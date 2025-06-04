from cleaners.duplicate_remove_cleaner import DuplicateRemoverCleaner
from cleaning_pipeline import CleaningPipeline
from fetchers.local_source_fetcher import LocalSourceFetcher



# An example registry structure
# registry = {
#     "source1": {
#         "fetcher": '',
#         "cleaner": '',
#     },
#     "source2": {
#         "fetcher": '',
#         "cleaner": ''
#     },
# }

local_fetcher = LocalSourceFetcher(file_path='/Users/orlevi/PWC/tmp/AllHebNLIFiles-Deduped-D2.forgpt_part-100_sample.csv',
                                   output_path='cleaned_data')

registry = {'source_test': {
                "fetcher": local_fetcher,
                "cleaner": DuplicateRemoverCleaner()
            }
}

def run_all():
    for source_name, components in registry.items():
        pipeline = CleaningPipeline(
            fetcher=components["fetcher"],
            cleaner=components["cleaner"],
            source_name=source_name
        )
        pipeline.run()

run_all()