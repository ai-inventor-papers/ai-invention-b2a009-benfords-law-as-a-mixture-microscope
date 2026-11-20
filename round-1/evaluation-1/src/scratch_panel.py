import json
from pathlib import Path
from loguru import logger
import panel

logger.remove()
logger.add(lambda m: print(m, end=""), level="INFO")
s = panel.build_panel(Path('data/owid_population_unwpp.csv'),
                      Path('data/wb_population_raw.json'),
                      Path('data/wb_countries.json'),
                      Path('data'))
print(json.dumps({k: v for k, v in s.items() if k not in ('owid', 'taglists', 'dropped_entities')}, indent=2)[:3000])
print('\nTAGLISTS:')
print(json.dumps(s['taglists'], indent=2)[:1800])
print('\nn dropped entities:', len(s['dropped_entities']))
