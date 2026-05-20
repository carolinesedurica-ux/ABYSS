import sys
from pathlib import Path

# Add root directory to sys.path so we can import dark_data_miner
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dark_data_miner.api.server import app
