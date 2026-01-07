import pandas as pd
import os
import csv


class Result:
    def __init__(self, schema):
        self.schema = schema
        self.rows = []

    def add_row(self, **kwargs):
        row = {col: kwargs.get(col, None) for col in self.schema}
        self.rows.append(row)

    def to_dataframe(self):
        if not self.rows:
            return pd.DataFrame(columns=self.schema)
        df = pd.DataFrame(self.rows)
        return df.reindex(columns=self.schema)

    def save_csv(self, filepath):
        df = self.to_dataframe()
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        df.to_csv(filepath, index=False)


class StreamingResult:
    """Memory-efficient Result class that writes directly to CSV."""

    def __init__(self, schema, filepath):
        self.schema = schema
        self.filepath = filepath
        self.csv_file = None
        self.csv_writer = None
        self._header_written = False

        # Ensure directory exists
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        # Open file and write header
        self.csv_file = open(filepath, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.schema)
        self.csv_writer.writeheader()
        self._header_written = True

    def add_row(self, **kwargs):
        """Add a row and immediately write to CSV."""
        row = {col: kwargs.get(col, None) for col in self.schema}
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written to disk

    def close(self):
        """Close the CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None