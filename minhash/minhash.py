from datatrove.executor import RayPipelineExecutor
import ray
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages

def main():
    # connect to ray cluster
    ray.init(address="auto")
    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )  # better precision -> fewer false positives (collisions)

    S3_MINHASH_BASE_PATH = "s3://gepeta-datasets/dedupe/run1"

    S3_LOGS_FOLDER = f"{S3_MINHASH_BASE_PATH}/logs"
    LOCAL_LOGS_FOLDER = "logs/minhash"

    TOTAL_TASKS = 1000

    # this is the original data that we want to deduplicate
    INPUT_READER = JsonlReader(
                data_folder="s3://gepeta-datasets/processed_cleaned_filtered/run_5_files",
                recursive=True,  # will traverse all source_name_x folders
                compression="gzip",
                glob_pattern="*.jsonl.gz",
            )

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = RayPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{S3_MINHASH_BASE_PATH}/signatures", config=minhash_config, language=Languages.hebrew__hebr
            ),
        ],
        tasks=TOTAL_TASKS,
        workers=-1,
        mem_per_cpu_gb=2,
        cpus_per_task=1,
        logging_dir=f"{S3_LOGS_FOLDER}/signatures",
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = RayPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
                output_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        logging_dir=f"{S3_LOGS_FOLDER}/buckets",
        depends=stage1,
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = RayPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
                output_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        logging_dir=f"{S3_LOGS_FOLDER}/clusters",
        mem_per_cpu_gb=30,
        cpus_per_task=4,
        workers=2,
        depends=stage2,
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = RayPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
            MinhashDedupFilter(
                input_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
                exclusion_writer=JsonlWriter(f"{S3_MINHASH_BASE_PATH}/removed"),
            ),
            JsonlWriter(output_folder=f"{S3_MINHASH_BASE_PATH}/deduplicated_output"),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{S3_LOGS_FOLDER}/filter",
        depends=stage3,
        workers=15,
        cpus_per_task=3,
        mem_per_cpu_gb=8,

    )


    stage4.run()



if __name__ == "__main__":
    main()