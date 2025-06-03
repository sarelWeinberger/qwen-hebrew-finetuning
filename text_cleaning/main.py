from text_cleaning.cleaning_pipeline import CleaningPipeline

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


registry = {}

def run_all():
    for source_name, components in registry.items():
        pipeline = CleaningPipeline(
            source_fetcher=components["fetcher"],
            cleaner=components["cleaner"],
            output_path=f"cleaned_{source_name}.csv"
        )
        pipeline.run()