"""HAR Spark Cloud package: distributed human-activity recognition with PySpark.

The package is organised as a pipeline: ``config`` loads YAML settings, ``spark``
creates the session, ``ingest`` parses WISDM rows, ``features`` builds and splits
window features, ``model`` trains/evaluates MLlib classifiers, and ``pipeline``
coordinates those stages.  ``cli`` exposes the workflow as the ``har-spark`` command.
Importing this package performs no Spark or file-system work.
"""

__version__ = "0.1.0"

