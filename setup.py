"""Setup file for dannce."""
from setuptools import setup, find_packages

setup(
    name="npmp_embed",
    version="1.0.0",
    packages=find_packages(),
    scripts=['multi_job_embed.sh', 'multi_job_embed_preprocessing.sh', 'embed.sh'],
    entry_points={
        "console_scripts": [
            "dispatch-npmp-embed = dispatch_embed:dispatch_npmp_embed",
            "npmp_embed_single_batch = embed:npmp_embed_single_batch",
            "npmp_embed = embed:npmp_embed", 
            "npmp-preprocessing = mocap_preprocess:submit",
            "parallel-npmp-preprocessing = mocap_preprocess:parallel_submit",
            "npmp-preprocessing-single-batch = mocap_preprocess:npmp_embed_preprocessing_single_batch",
            "merge-npmp-preprocessing = mocap_preprocess:merge_preprocessed_files",
            "merge-embed = merge_embed:merge_files",
            "render_training_set_single_batch = generate_training_data:render_training_set_single_batch",
        ]
    },
)
