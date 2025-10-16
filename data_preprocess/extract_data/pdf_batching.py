from __future__ import print_function, division
from multiprocessing import Pool, cpu_count
from google.cloud import storage
from google.cloud.storage import  transfer_manager
from logging.handlers import RotatingFileHandler
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

import argparse
import boto3
import fitz
import itertools
import json
import logging
import numpy as np
import os
import re
import sys
import time

load_dotenv()

mytime = time.clock if str is bytes else time.perf_counter
white_threshold = 245 # value of white at least
empty_threshold = 0.999 # how many white pixels need to be for empty page recognition
dpi = 150

# needs instanciating for uploading anyway
storage_client = storage.Client(project=os.getenv("PROJECT_ID"))
bucket = storage_client.bucket(os.getenv("BUCKET_NAME"))

s3 = boto3.client("s3")

def setup_logger():
    # Silence all existing loggers first
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL)

    # Silence noisy third-party libraries
    logging.getLogger('google').setLevel(logging.CRITICAL)
    logging.getLogger('google.cloud').setLevel(logging.CRITICAL)
    logging.getLogger('google.auth').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('requests').setLevel(logging.CRITICAL)
    logging.getLogger('PIL').setLevel(logging.CRITICAL)
    logging.getLogger('fitz').setLevel(logging.CRITICAL)

    logger = logging.getLogger("pdf_processor")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't send to root logger

    # Create rotating file handler (max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(
        './logs/pdf_processing.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )

    # Create console handler for real-time output
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

def process_pdf_chunk(vector):
    """Render a page range of a document.

    Notes:
        The PyMuPDF document cannot be part of the argument, because that
        cannot be pickled. So we are being passed in just its filename.
        This is no performance issue, because we are a separate process and
        need to open the document anyway.
        Any page-specific function can be processed here - rendering is just
        an example - text extraction might be another.
        The work must however be self-contained: no inter-process communication
        or synchronization is possible with this design.
        Care must also be taken with which parameters are contained in the
        argument, because it will be passed in via pickling by the Pool class.
        So any large objects will increase the overall duration.
    Args:
        vector: a list containing required parameters.
    """
    # recreate the arguments
    idx = vector[0]  # this is the segment number we have to process
    cpu = vector[1]  # number of CPUs
    filename = vector[2]  # document filename
    if isinstance(filename, str):
        doc = fitz.open(filename)  # open the document
    else:
        doc = fitz.open(stream=filename)
    num_pages = doc.page_count  # get number of pages

    # pages per segment: make sure that cpu * seg_size >= num_pages!
    seg_size = int(num_pages / cpu + 1)
    seg_from = idx * seg_size  # our first page number
    seg_to = min(seg_from + seg_size, num_pages)  # last page number

    results = []

    for i in range(seg_from, seg_to):  # work through our page segment
        page = doc[i]
        img = page2img(page, dpi)
        is_empty, white_percent = count_pixels(img, white_threshold, empty_threshold)
        if not is_empty:
            tmp_file_name = f"/tmp/chunk_{i}.png"
            img.save(tmp_file_name)        
        page_data = {"page": i,
                    "img_path": tmp_file_name if not is_empty else None,
                    "is_empty": is_empty,
                    "white_percent": white_percent}
        results.append(page_data)
    logger.info(f"Processed page numbers {seg_from} through {seg_to-1}")
    return results


def process_pdf(doc_path):
    t0 = mytime()  # start a timer
    cpu = cpu_count() - 1

    # make vectors of arguments for the processes
    vectors = [(i, cpu, doc_path) for i in range(cpu)]
    logger.info(f"Starting {cpu} processes.")

    pool = Pool(processes=cpu)  # make pool of 'cpu_count()' processes
    results = pool.map(process_pdf_chunk, vectors, 1)  # start processes passing each a vector

    t1 = mytime()  # stop the timer
    logger.info(f"Total time {round(t1 - t0, 2)} seconds")

    results = list(itertools.chain.from_iterable(results))
    
    return results


def page2img(page, dpi=150):
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    
    # Convert to PIL Image then to numpy array
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    gray_img = img.convert('L')

    return gray_img

def count_pixels(img, white_threshold=240, empty_threshold=0.9):
    pixels = np.array(img)
    # Count white pixels
    white_pixels = np.sum(pixels >= white_threshold)
    total_pixels = pixels.size
    white_percentage = white_pixels / total_pixels
    
    is_empty = white_percentage >= empty_threshold

    return is_empty.item(), white_percentage


def gcp_route(prefix):
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in tqdm(blobs):
        blob_name = blob.name.split("/")[-1].strip(".pdf")

        # check if already chunked
        res_blobs = bucket.list_blobs(prefix=f"data_HQwOcr_img_chunks/{blob_name}", max_results=1)
        if len(list(res_blobs)) > 0:
            continue
        
        full_pdf = blob.download_as_bytes()
        logger.info(f"Processing {blob.name}")
        results = process_pdf(full_pdf)
        filenames = []
        for res in results:
            if res["img_path"]:
                filenames.append(res["img_path"].strip("/tmp/"))
            else:
                logger.info(f"{blob_name}, chunk {res["page"]} is EMPTY ({res["white_percent"]})")

        transfer_manager_results = transfer_manager.upload_many_from_filenames(
                bucket, filenames, source_directory="/tmp", threads=2, blob_name_prefix=f"data_HQwOcr_img_chunks/{blob_name}/{blob_name}_"
            )

        for name, res in zip(filenames, transfer_manager_results):
            # The results list is either `None` or an exception for each filename in
            # the input list, in order.

            if isinstance(res, Exception):
                logger.info("Failed to upload {} due to exception: {}".format(name, res))
            else:
                logger.info("Uploaded {} to {}.".format(name, bucket.name))
        
        for fn in filenames:
            tmp_fn = f"/tmp/{fn}"
            try:
                os.remove(tmp_fn)
                logger.info(f"Successfully deleted: {tmp_fn}")
            except FileNotFoundError:
                logger.warning(f"File not found: {tmp_fn}")
            except Exception as e:
                logger.error(f"Error deleting {tmp_fn}: {e}")


def aws_route(blob_key):
    blob = None
    try:
        response = s3.list_objects_v2(Bucket=os.getenv("AWS_BUCKET_NAME"), Prefix=blob_key)
        if "Contents" in response:
            for tmp in response["Contents"]:
                if tmp["Key"].lower().endswith(".pdf"):
                    blob = s3.get_object(Bucket=os.getenv("AWS_BUCKET_NAME"), Key=tmp["Key"])
        else:
            logger.info(f"Couldn't find {blob_key}")
            return
        if not blob:
            logger.info(f"Couldn't find {blob_key}")
            return
    except Exception as e:
        logger.info(f"Error: {e}")
        return

    blob_name = blob_key.split("/")[-1].strip(".pdf")
    # check if already chunked
    res_blobs = bucket.list_blobs(prefix=f"data_HQ_wOcr_img_chunks/{blob_name}", max_results=1)
    if len(list(res_blobs)) > 0:
        logger.info(f"Skipping {blob_key} as its chunks are already in gcp")
        return

    # Downloading the pdf
    full_pdf = blob["Body"].read()
    logger.info(f"Processing {blob_key}")

    # processing the pdf
    results = process_pdf(full_pdf)
    filenames = []
    for res in results:
        if res["img_path"]:
            filenames.append(res["img_path"].strip("/tmp/"))
        else:
            logger.info(f"{blob_name}, chunk {res["page"]} is EMPTY ({res["white_percent"]})")

    # uploading the images
    transfer_manager_results = transfer_manager.upload_many_from_filenames(
            bucket, filenames, source_directory="/tmp", threads=2, blob_name_prefix=f"data_HQ_wOcr_img_chunks/{blob_name}/{blob_name}_"
        )

    # logging the upload results
    for name, res in zip(filenames, transfer_manager_results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(res, Exception):
            logger.info("Failed to upload {} due to exception: {}".format(name, res))
        else:
            logger.info("Uploaded {} to {}.".format(name, bucket.name))

    # deleting the temporary files
    for fn in filenames:
        tmp_fn = f"/tmp/{fn}"
        try:
            os.remove(tmp_fn)
            logger.info(f"Successfully deleted: {tmp_fn}")
        except FileNotFoundError:
            logger.warning(f"File not found: {tmp_fn}")
        except Exception as e:
            logger.error(f"Error deleting {tmp_fn}: {e}")
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process PDF files from GCS (or AWS) bucket')
    parser.add_argument('--prefix', default='data/HQ', 
                       help='Blob prefix to process (default: "data/HQ")')
    parser.add_argument('--source', default="gcp", help="Source cloud storage (gcp or aws)")
    parser.add_argument('--fnlist', help="List of file names to filter and process")
    args = parser.parse_args()

    fnlist = None
    if args.fnlist:
        with open(args.fnlist, "r", encoding="utf-8") as f:
            fnlist = f.read()
            try:
                fnlist = json.loads(fnlist)
            except:
                print("fnlist needs to lead to a LIST")
                fnlist = None

    if args.source == "gcp":
        gcp_route(args.prefix)
    elif args.source == "aws":
        if fnlist:
            for fn in tqdm(fnlist):
                blob_key = f"{args.prefix}/{fn}"
                aws_route(blob_key)
        else:
            print("Not implemented yet")
    else:
        print("No valid source cloud storage provided")