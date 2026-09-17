import os
from supabase import create_client, Client

_url = os.environ["SUPABASE_URL"]
_key = os.environ["SUPABASE_SERVICE_KEY"]   # service key needed for inserts from a backend script

supabase: Client = create_client(_url, _key)