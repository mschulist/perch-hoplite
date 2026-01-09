"""DuckDB database implementation using USearch for vector storage & search."""

import collections
from collections.abc import Sequence
import dataclasses
import datetime as dt
import json
import duckdb
from typing import Any

from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import interface
from perch_hoplite.db import sqlite_usearch_impl
from sqlite_usearch_impl import (
    deserialize_array,
    is_valid_sql_identifier,
    format_sql_insert_values,
    format_sql_where_conditions,
)
from usearch import index as uindex


DUCKDB_FILENAME = "hoplite.duckdb"


@dataclasses.dataclass
class DuckDBUSearchDB(interface.HopliteDBInterface):
    """DuckDB hoplite database, using USearch for vector storage.

    USearch provides both indexing for approximate nearest neighbor search and
    fast disk-based random access to vectors for the complete database. USearch
    will default to working with disk-based vectors, unless we insert or remove
    embeddings, in which case we load the index into memory and use it from there
    for all subsequent operations. On database commit, the in-memory index is
    persisted to disk.

    Attributes:
      db_path: The path to the database directory.
      connection: The duckdb database connection.
      ui: The USearch index.
      _embedding_dim: The dimension of the embeddings.
      _embedding_dtype: The data type of the embeddings.
      _ui_loaded: Whether the USearch index was loaded in memory.
      _ui_updated: Whether the USearch index was updated since the last load and
        needs to be persisted to disk at some point in the future.
    """

    # User-provided.
    db_path: epath.Path

    # Instantiated during creation.
    connection: duckdb.DuckDBPyConnection
    ui: uindex.Index

    # Obtained from `usearch_cfg`.
    _embedding_dim: int
    _embedding_dtype: type[Any] = np.float16

    # Dynamic state.
    _extra_table_columns: dict[str, dict[str, type[Any]]] = dataclasses.field(
        default_factory=lambda: {
            "deployments": {},
            "recordings": {},
            "windows": {},
            "annotations": {},
        }
    )
    _ui_loaded: bool = False
    _ui_updated: bool = False

    @property
    def duckdb_path(self) -> epath.Path:
        return self.db_path / DUCKDB_FILENAME

    @property
    def usearch_path(self) -> epath.Path:
        return self.db_path / sqlite_usearch_impl.UINDEX_FILENAME

    @staticmethod
    def _setup_tables(connection: duckdb.DuckDBPyConnection) -> None:
        """Create the DuckDB tables.

        Args:
        connection: The DuckDB connection to use.
        """

        # Skip setting up the tables if they already exist.
        connection.execute("""
        SELECT name
        FROM sqlite_master
        WHERE name = "windows" AND type = "table"
        """)
        if connection.fetchone() is not None:
            return

        # Create the deployments table.
        connection.execute("""
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            project TEXT NOT NULL,
            latitude REAL,
            longitude REAL
        )
        """)

        # Create the recordings table.
        connection.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            datetime TEXT,
            deployment_id INTEGER,
            FOREIGN KEY (deployment_id) REFERENCES deployments(id)
            ON DELETE CASCADE
        )
        """)

        # Create the windows table.
        connection.execute("""
        CREATE TABLE IF NOT EXISTS windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL,
            offsets BLOB NOT NULL,
            FOREIGN KEY (recording_id) REFERENCES recordings(id)
            ON DELETE CASCADE
        )
        """)

        # Create the annotations table.
        connection.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            window_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            label_type INTEGER NOT NULL,
            provenance TEXT NOT NULL,
            FOREIGN KEY (window_id) REFERENCES windows(id)
            ON DELETE CASCADE
        )
        """)

        # Create the metadata table.
        connection.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

        # Create indexes for efficient lookups.
        connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_deployments
        ON deployments(id)
        """)
        connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_recordings
        ON recordings(id, deployment_id)
        """)
        connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_windows
        ON windows(id, recording_id)
        """)
        connection.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_annotations
        ON annotations(id, window_id)
        """)
        connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels
        ON annotations(window_id, label, label_type, provenance)
        """)

    @staticmethod
    def _get_all_metadata(
        connection: duckdb.DuckDBPyConnection,
    ) -> config_dict.ConfigDict:
        """Get all key-value pairs from the metadata table.

        Args:
          connection: The DuckDB connection to use.

        Returns:
          A ConfigDict containing all the metadata.
        """

        result = connection.execute("""
        SELECT key, value
        FROM hoplite_metadata
        """).fetchall()
        return config_dict.ConfigDict({k: json.loads(v) for k, v in result})

    @classmethod
    def create(
        cls,
        db_path: str | epath.Path,
        usearch_cfg: config_dict.ConfigDict | None = None,
    ) -> "DuckDBUSearchDB":
        """Connect to and, if needed, initialize the database.

        Args:
          db_path: The path to the database directory.
          usearch_cfg: The configuration for the USearch index. If None, the config
            is loaded from the DB.

        Returns:
          A new instance of the database.
        """

        # Create the DuckDB DB.
        db_path = epath.Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        duckdb_path = db_path / DUCKDB_FILENAME
        connection = duckdb.connect(duckdb_path.as_posix())
        cls._setup_tables(connection)
        connection.commit()

        # Retrieve the metadata.
        # TODO(tomdenton): Check that `usearch_cfg` is consistent with the DB.
        metadata = cls._get_all_metadata(connection)
        if (
            sqlite_usearch_impl.USEARCH_CONFIG_KEY in metadata
            and usearch_cfg is not None
            and metadata[sqlite_usearch_impl.USEARCH_CONFIG_KEY] != usearch_cfg
        ):
            raise ValueError(
                "A usearch_cfg was provided, but a different one already exists in"
                " the DB."
            )
        if sqlite_usearch_impl.USEARCH_CONFIG_KEY in metadata:
            usearch_cfg = metadata[sqlite_usearch_impl.USEARCH_CONFIG_KEY]  # type: ignore
        elif usearch_cfg is None:
            raise ValueError("No usearch_cfg was found in DB and none was provided.")

        # Create the USearch index.
        usearch_dtype = sqlite_usearch_impl.USEARCH_DTYPES[usearch_cfg.dtype]  # type: ignore
        index_path = db_path / sqlite_usearch_impl.UINDEX_FILENAME
        if index_path.exists():
            ui = uindex.Index(
                ndim=usearch_cfg.embedding_dim,  # type: ignore
                path=index_path,
                view=True,
            )
            ui_in_memory = False
        else:
            ui = uindex.Index(
                ndim=usearch_cfg.embedding_dim,  # type: ignore
                metric=getattr(uindex.MetricKind, usearch_cfg.metric_name),  # type: ignore
                expansion_add=usearch_cfg.expansion_add,  # type: ignore
                expansion_search=usearch_cfg.expansion_search,  # type: ignore
                dtype=usearch_dtype,
                path=index_path,
                view=False,
            )
            ui_in_memory = True

        # Create the Hoplite DB.
        hoplite_db = cls(
            db_path=db_path,
            connection=connection,
            ui=ui,
            _embedding_dim=usearch_cfg.embedding_dim,  # type: ignore
            _embedding_dtype=usearch_cfg.dtype,  # type: ignore
            _ui_loaded=ui_in_memory,
            _ui_updated=ui_in_memory,
        )
        hoplite_db.insert_metadata(sqlite_usearch_impl.USEARCH_CONFIG_KEY, usearch_cfg)  # type: ignore
        hoplite_db.commit()
        return hoplite_db

    def add_extra_table_column(
        self,
        table_name: str,
        column_name: str,
        column_type: type[Any],
    ) -> None:
        """Add an extra column to a table in the database."""

        if table_name not in [
            "deployments",
            "recordings",
            "windows",
            "annotations",
        ]:
            raise ValueError(f"Table `{table_name}` does not exist.")
        if not is_valid_sql_identifier(column_name):
            raise ValueError(f"Column `{column_name}` is not a valid SQL identifier.")
        if not isinstance(column_type, type):
            raise ValueError(f"Column type `{column_type}` must be a type.")

        sql_type = {
            int: "INTEGER",
            float: "REAL",
            str: "TEXT",
            bytes: "BLOB",
            dt.datetime: "TEXT",
        }
        if column_type not in sql_type:
            raise ValueError(
                f"Column type `{column_type.__name__}` is not supported. Use one of:"
                f" {', '.join([key.__name__ for key in sql_type.keys()])}"
            )

        self.connection.execute(f"""
        ALTER TABLE {table_name}
        ADD COLUMN {column_name} {sql_type[column_type]}
        """)

        self._extra_table_columns[table_name][column_name] = column_type

    def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
        """Get all extra columns in the database."""
        return self._extra_table_columns

    def commit(self) -> None:
        """Commit any pending transactions to the database."""
        self.connection.commit()
        if self._ui_updated:
            self.ui.save()
            self._ui_updated = False

    def thread_split(self) -> "DuckDBUSearchDB":
        """Get a new instance of the DuckDB DB."""
        return self.create(self.db_path.as_posix())

    def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
        """Insert a key-value pair into the metadata table."""
        json_coded = value.to_json()
        self.connection.execute(
            """
        INSERT INTO hoplite_metadata (key, value)
        VALUES ($1, $2)
        ON CONFLICT (key)
        DO UPDATE SET value = excluded.value
        """,
            (key, json_coded),
        )

    def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
        """Get a key-value pair from the metadata table."""

        if key is None:
            result = self.connection.execute("""
          SELECT key, value
          FROM hoplite_metadata
          """).fetchall()
            return config_dict.ConfigDict({k: json.loads(v) for k, v in result})

        result = self.connection.execute(
            """
        SELECT value
        FROM hoplite_metadata
        WHERE key = $1
        """,
            (key,),
        ).fetchone()
        if result is None:
            raise KeyError(f"Metadata key not found: {key}")
        return config_dict.ConfigDict(json.loads(result[0]))

    def remove_metadata(self, key: str | None) -> None:
        """Remove a key-value pair from the metadata table."""

        if key is None:
            self.connection.execute("""
          DELETE FROM hoplite_metadata
          """)
            return

        result = self.connection.execute(
            """
        DELETE FROM hoplite_metadata
        WHERE key = $1
        """,
            (key,),
        )
        if result.fetchone() is None:
            raise KeyError(f"Metadata key not found: {key}")

    def insert_deployment(
        self,
        name: str,
        project: str,
        latitude: float | None = None,
        longitude: float | None = None,
        **kwargs: Any,
    ) -> int:
        """Insert a deployment into the database."""
        columns_str, placeholders_str, values = format_sql_insert_values(
            name=name,
            project=project,
            latitude=latitude,
            longitude=longitude,
            **kwargs,
        )
        self.connection.execute(
            f"""
        INSERT INTO deployments {columns_str}
        VALUES {placeholders_str}
        """,
            values,
        )

        # Get the last inserted id from the sequence
        result = self.connection.execute("SELECT currval('deployments_seq')").fetchone()
        if result is None:
            raise RuntimeError("Error getting deployment ID from the database.")
        return result[0]

    def get_deployment(self, deployment_id: int) -> interface.Deployment:
        """Get a deployment from the database."""
        result = self.connection.execute(
            """
        SELECT *
        FROM deployments
        WHERE id = $1
        """,
            (deployment_id,),
        ).fetchone()
        if result is None:
            raise KeyError(f"Deployment id not found: {deployment_id}")

        columns = [desc[0] for desc in self.connection.description]
        return interface.Deployment(**dict(zip(columns, result)))

    def remove_deployment(self, deployment_id: int) -> None:
        """Remove a deployment from the database."""
        # TODO(stefanistrate): Make sure to remove corresponding embeddings from
        # USearch if removing this deployment triggers window removals.
        # First check if the deployment exists
        check = self.connection.execute(
            "SELECT id FROM deployments WHERE id = $1",
            (deployment_id,),
        ).fetchone()
        if check is None:
            raise KeyError(f"Deployment id not found: {deployment_id}")

        self.connection.execute(
            """
        DELETE FROM deployments
        WHERE id = $1
        """,
            (deployment_id,),
        )

    def insert_recording(
        self,
        filename: str,
        datetime: dt.datetime | None = None,
        deployment_id: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Insert a recording into the database."""
        columns_str, placeholders_str, values = format_sql_insert_values(
            filename=filename,
            datetime=datetime,
            deployment_id=deployment_id,
            **kwargs,
        )
        self.connection.execute(
            f"""
        INSERT INTO recordings {columns_str}
        VALUES {placeholders_str}
        """,
            values,
        )

        result = self.connection.execute("SELECT currval('recordings_seq')").fetchone()
        if result is None:
            raise RuntimeError("Error getting recording ID from the database.")
        return result[0]

    def get_recording(self, recording_id: int) -> interface.Recording:
        """Get a recording from the database."""
        result = self.connection.execute(
            """
        SELECT *
        FROM recordings
        WHERE id = $1
        """,
            (recording_id,),
        ).fetchone()
        if result is None:
            raise KeyError(f"Recording id not found: {recording_id}")

        columns = [desc[0] for desc in self.connection.description]
        recording = interface.Recording(**dict(zip(columns, result)))
        if recording.datetime is not None and isinstance(recording.datetime, str):
            recording.datetime = dt.datetime.fromisoformat(recording.datetime)
        return recording

    def remove_recording(self, recording_id: int) -> None:
        """Remove a recording from the database."""
        # TODO(stefanistrate): Make sure to remove corresponding embeddings from
        # USearch if removing this recording triggers window removals.
        # First check if the recording exists
        check = self.connection.execute(
            "SELECT id FROM recordings WHERE id = $1",
            (recording_id,),
        ).fetchone()
        if check is None:
            raise KeyError(f"Recording id not found: {recording_id}")

        self.connection.execute(
            """
        DELETE FROM recordings
        WHERE id = $1
        """,
            (recording_id,),
        )

    def insert_window(
        self,
        recording_id: int,
        offsets: np.ndarray,
        embedding: np.ndarray | None = None,
        **kwargs: Any,
    ) -> int:
        """Insert a window into the database."""

        if embedding is not None and embedding.shape[-1] != self._embedding_dim:
            raise ValueError(
                f"Incorrect embedding dimension. Expected {self._embedding_dim}, but"
                f" got {embedding.shape[-1]}."
            )

        columns_str, placeholders_str, values = format_sql_insert_values(
            recording_id=recording_id,
            offsets=offsets,
            **kwargs,
        )
        self.connection.execute(
            f"""
        INSERT INTO windows {columns_str}
        VALUES {placeholders_str}
        """,
            values,
        )

        result = self.connection.execute("SELECT currval('windows_seq')").fetchone()
        if result is None:
            raise RuntimeError("Error getting window ID from the database.")
        window_id = result[0]
        if embedding is not None:
            if not self._ui_loaded:
                self.ui.load()
                self._ui_loaded = True
            self.ui.add(window_id, embedding.astype(self._embedding_dtype))
            self._ui_updated = True
        return window_id

    def get_window(
        self,
        window_id: int,
        include_embedding: bool = False,
    ) -> interface.Window:
        """Get a window from the database."""

        result = self.connection.execute(
            """
        SELECT *
        FROM windows
        WHERE id = $1
        """,
            (window_id,),
        ).fetchone()
        if result is None:
            raise KeyError(f"Window id not found: {window_id}")

        columns = [desc[0] for desc in self.connection.description]
        window = interface.Window(
            embedding=None,
            **dict(zip(columns, result)),
        )
        if isinstance(window.offsets, bytes):
            window.offsets = deserialize_array(window.offsets, np.float32)
        if include_embedding:
            window.embedding = self.get_embedding(window_id)
        return window

    def get_embedding(self, window_id: int) -> np.ndarray:
        """Get an embedding vector from the database."""

        found = self.ui.contains(window_id)
        if not isinstance(found, bool):
            raise RuntimeError(
                "Expected bool result from the USearch `contains()` method, but got"
                f" {type(found)}: {found}."
            )
        if not found:
            raise KeyError(f"Embedding vector not found for window id: {window_id}")
        embedding = self.ui.get(window_id)
        if not isinstance(embedding, np.ndarray):
            raise RuntimeError(
                "Expected np.ndarray result from the USearch `get()` method, but got"
                f" {type(embedding)}: {embedding}."
            )
        return embedding

    def get_embeddings_batch(
        self,
        window_ids: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        """Get a batch of embedding vectors from the database."""

        if not isinstance(window_ids, np.ndarray):
            window_ids = np.array(window_ids, dtype=np.int32)

        found = self.ui.contains(window_ids)
        if not isinstance(found, np.ndarray):
            raise RuntimeError(
                "Expected np.ndarray result from the USearch `contains()` method, but"
                f" got {type(found)}: {found}."
            )
        if not np.all(found):
            raise KeyError(
                "Embedding vectors not found for window ids:"
                f" {window_ids[~found].tolist()}"
            )
        embeddings_batch = self.ui.get(window_ids)
        if not isinstance(embeddings_batch, np.ndarray):
            raise RuntimeError(
                "Expected np.ndarray result from the USearch `get()` method, but got"
                f" {type(embeddings_batch)}: {embeddings_batch}."
            )
        return embeddings_batch

    def remove_window(self, window_id: int) -> None:
        """Remove a window from the database."""

        # First check if the window exists
        check = self.connection.execute(
            "SELECT id FROM windows WHERE id = $1",
            (window_id,),
        ).fetchone()
        if check is None:
            raise KeyError(f"Window id not found: {window_id}")

        self.connection.execute(
            """
        DELETE FROM windows
        WHERE id = $1
        """,
            (window_id,),
        )

        if not self._ui_loaded:
            self.ui.load()
            self._ui_loaded = True
        self.ui.remove(window_id)
        self._ui_updated = True

    def insert_annotation(
        self,
        window_id: int,
        label: str,
        label_type: interface.LabelType,
        provenance: str,
        skip_duplicates: bool = False,
        **kwargs: Any,
    ) -> int:
        """Insert an annotation into the database."""

        if skip_duplicates:
            matches = self.get_all_annotations(
                filter=config_dict.create(
                    eq=dict(window_id=window_id, label=label, label_type=label_type)
                )
            )
            if matches:
                return matches[0].id

        columns_str, placeholders_str, values = format_sql_insert_values(
            window_id=window_id,
            label=label,
            label_type=label_type,
            provenance=provenance,
            **kwargs,
        )
        self.connection.execute(
            f"""
        INSERT INTO annotations {columns_str}
        VALUES {placeholders_str}
        """,
            values,
        )

        result = self.connection.execute("SELECT currval('annotations_seq')").fetchone()
        if result is None:
            raise RuntimeError("Error getting annotation ID from the database.")
        return result[0]

    def get_annotation(self, annotation_id: int) -> interface.Annotation:
        """Get an annotation from the database."""

        result = self.connection.execute(
            """
        SELECT *
        FROM annotations
        WHERE id = $1
        """,
            (annotation_id,),
        ).fetchone()
        if result is None:
            raise KeyError(f"Annotation id not found: {annotation_id}")

        columns = [desc[0] for desc in self.connection.description]
        annotation = interface.Annotation(**dict(zip(columns, result)))
        annotation.label_type = interface.LabelType(annotation.label_type)
        return annotation

    def remove_annotation(self, annotation_id: int) -> None:
        """Remove an annotation from the database."""

        # First check if the annotation exists
        check = self.connection.execute(
            "SELECT id FROM annotations WHERE id = $1",
            (annotation_id,),
        ).fetchone()
        if check is None:
            raise KeyError(f"Annotation id not found: {annotation_id}")

        self.connection.execute(
            """
        DELETE FROM annotations
        WHERE id = $1
        """,
            (annotation_id,),
        )

    def count_embeddings(self) -> int:
        """Get the number of embeddings in the database."""
        return self.ui.size

    def match_window_ids(
        self,
        deployments_filter: config_dict.ConfigDict | None = None,
        recordings_filter: config_dict.ConfigDict | None = None,
        windows_filter: config_dict.ConfigDict | None = None,
        annotations_filter: config_dict.ConfigDict | None = None,
        limit: int | None = None,
    ) -> Sequence[int]:
        """Get matching window IDs from the database based on given filters."""

        # TODO(stefanistrate): Is it more efficient for large databases to run
        # multiple SELECT queries (in order: by deployment, by recording, by window,
        # by annotation) instead of running one giant JOIN?

        # Pick which tables need to be queried.
        query_tables = set()
        if annotations_filter:
            query_tables |= {"annotations"}
        if windows_filter:
            query_tables |= {"windows"}
        if recordings_filter:
            query_tables |= {"windows", "recordings"}
        if deployments_filter:
            query_tables |= {"windows", "recordings", "deployments"}
        if not any(
            [
                annotations_filter,
                windows_filter,
                recordings_filter,
                deployments_filter,
            ]
        ):
            query_tables = {"windows"}

        # Build the `SELECT ... FROM ... [JOIN ...]` part of the SQL query.
        if "annotations" in query_tables:
            select_clause = "SELECT DISTINCT annotations.window_id"
            from_clause = "FROM annotations"
        else:
            select_clause = "SELECT windows.id"
            from_clause = "FROM windows"
        if "annotations" in query_tables and "windows" in query_tables:
            from_clause += " JOIN windows ON annotations.window_id = windows.id"
        if "windows" in query_tables and "recordings" in query_tables:
            from_clause += " JOIN recordings ON windows.recording_id = recordings.id"
        if "recordings" in query_tables and "deployments" in query_tables:
            from_clause += (
                " JOIN deployments ON recordings.deployment_id = deployments.id"
            )

        # Build the `WHERE ...` part of the SQL query.
        # Need to track param counter across all conditions
        all_conditions = []
        all_values = []

        for filter_dict, table_prefix in [
            (annotations_filter, "annotations"),
            (windows_filter, "windows"),
            (recordings_filter, "recordings"),
            (deployments_filter, "deployments"),
        ]:
            cond_str, cond_values = format_sql_where_conditions(
                filter_dict, table_prefix=table_prefix
            )
            if cond_str:
                all_conditions.append(cond_str)
                all_values.extend(cond_values)

        conditions_str = " AND ".join(all_conditions)
        where_clause = f"WHERE {conditions_str}" if conditions_str else ""

        # Build the `LIMIT ...` part of the SQL query.
        if limit is None:
            limit_clause = ""
        else:
            limit_clause = f"LIMIT {limit}"

        # Execute the SQL query and return the window IDs.
        result = self.connection.execute(
            f"""
        {select_clause}
        {from_clause}
        {where_clause}
        {limit_clause}
        """,
            all_values,
        ).fetchall()
        return [row[0] for row in result]

    def get_all_projects(self) -> Sequence[str]:
        """Get all distinct projects from the database."""

        result = self.connection.execute("""
        SELECT DISTINCT project
        FROM deployments
        ORDER BY project
        """).fetchall()
        return [row[0] for row in result]

    def get_all_deployments(
        self,
        filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
    ) -> Sequence[interface.Deployment]:
        """Get all deployments from the database."""

        conditions_str, values = format_sql_where_conditions(filter)
        where_clause = f"WHERE {conditions_str}" if conditions_str else ""
        result = self.connection.execute(
            f"""
        SELECT *
        FROM deployments
        {where_clause}
        """,
            values,
        )

        columns = [desc[0] for desc in result.description]
        deployments = [
            interface.Deployment(**dict(zip(columns, row))) for row in result.fetchall()
        ]
        return deployments

    def get_all_recordings(
        self,
        filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
    ) -> Sequence[interface.Recording]:
        """Get all recordings from the database."""

        conditions_str, values = format_sql_where_conditions(filter)
        where_clause = f"WHERE {conditions_str}" if conditions_str else ""
        result = self.connection.execute(
            f"""
        SELECT *
        FROM recordings
        {where_clause}
        """,
            values,
        )

        recordings = []
        columns = [desc[0] for desc in result.description]
        for row in result.fetchall():
            recording = interface.Recording(**dict(zip(columns, row)))
            if recording.datetime is not None and isinstance(recording.datetime, str):
                recording.datetime = dt.datetime.fromisoformat(recording.datetime)
            recordings.append(recording)
        return recordings

    def get_all_windows(
        self,
        include_embedding: bool = False,
        filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
    ) -> Sequence[interface.Window]:
        """Get all windows from the database."""

        conditions_str, values = format_sql_where_conditions(filter)
        where_clause = f"WHERE {conditions_str}" if conditions_str else ""
        result = self.connection.execute(
            f"""
        SELECT *
        FROM windows
        {where_clause}
        """,
            values,
        )

        windows = []
        columns = [desc[0] for desc in result.description]
        for row in result.fetchall():
            window = interface.Window(
                embedding=None,
                **dict(zip(columns, row)),
            )
            if isinstance(window.offsets, bytes):
                window.offsets = deserialize_array(window.offsets, np.float32)
            if include_embedding:
                window.embedding = self.get_embedding(window.id)
            windows.append(window)
        return windows

    def get_all_annotations(
        self,
        filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
    ) -> Sequence[interface.Annotation]:
        """Get all annotations from the database."""

        conditions_str, values = format_sql_where_conditions(filter)
        where_clause = f"WHERE {conditions_str}" if conditions_str else ""
        result = self.connection.execute(
            f"""
        SELECT *
        FROM annotations
        {where_clause}
        """,
            values,
        )

        annotations = []
        columns = [desc[0] for desc in result.description]
        for row in result.fetchall():
            annotation = interface.Annotation(**dict(zip(columns, row)))
            annotation.label_type = interface.LabelType(annotation.label_type)
            annotations.append(annotation)
        return annotations

    def get_all_labels(
        self,
        label_type: interface.LabelType | None = None,
    ) -> Sequence[str]:
        """Get all distinct labels from the database."""

        if label_type is None:
            where_clause = ""
            values = []
        else:
            filter_dict = config_dict.create(eq=dict(label_type=label_type))
            conditions_str, values = format_sql_where_conditions(filter_dict)
            where_clause = f"WHERE {conditions_str}" if conditions_str else ""
        result = self.connection.execute(
            f"""
        SELECT DISTINCT label
        FROM annotations
        {where_clause}
        ORDER BY label
        """,
            values,
        ).fetchall()
        return [row[0] for row in result]

    def count_each_label(
        self,
        label_type: interface.LabelType | None = None,
    ) -> collections.Counter[str]:
        """Count each label in the database, ignoring provenance."""

        if label_type is None:
            where_clause = ""
            values = []
        else:
            filter_dict = config_dict.create(eq=dict(label_type=label_type))
            conditions_str, values = format_sql_where_conditions(filter_dict)
            where_clause = f"WHERE {conditions_str}" if conditions_str else ""

        # Subselect with DISTINCT is needed to avoid double-counting the same label
        # on the same window because of different provenances.
        result = self.connection.execute(
            f"""
        SELECT label, COUNT(*)
        FROM (
            SELECT DISTINCT window_id, label, label_type
            FROM annotations
            {where_clause}
        )
        GROUP BY label
        ORDER BY label
        """,
            values,
        ).fetchall()
        return collections.Counter({row[0]: row[1] for row in result})

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def get_embedding_dtype(self) -> type[Any]:
        """Get the embedding data type."""
        return self._embedding_dtype
