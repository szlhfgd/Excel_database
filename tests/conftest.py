import os

# Force every test to use an in-memory database. Without this, test modules
# that import `db`/`app` at load time would fall back to the default on-disk
# "spreadsheet.db" (the file the running app uses) and pollute it with test
# tables that reappear on the next test run.
os.environ["SPREADSHEET_DB"] = ":memory:"
