"""Quick polling check — once auth is set up, this exits 0 and prints status."""
import os, sys
from huggingface_hub import HfApi
api = HfApi()
try:
    me = api.whoami()
    info = api.model_info('facebook/sam3.1', token=True)
    print(f'AUTH OK — user={me.get("name")}, sam3.1 reachable, gated={info.gated}')
    sys.exit(0)
except Exception as e:
    print(f'NOT READY: {type(e).__name__}: {str(e)[:200]}')
    sys.exit(1)
