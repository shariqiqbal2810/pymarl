REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .icql_runner import ICQLEpisodeRunner
REGISTRY["icql"] = ICQLEpisodeRunner

from .icql_runner import ICQLParallelRunner
REGISTRY["icql_parallel"] = ICQLParallelRunner
