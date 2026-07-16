# The magnetism data is obtained from Materials Project API using the mp-api package.
# mp-1228868 number of atoms: 2, formula: NiAl, space group: Pm-3m, magnetism: non-magnetic
from mp_api.client import MPRester

with MPRester(api_key="<enter your api key>") as mpr:
    magnetism_doc = mpr.magnetism.search(material_ids=["mp-1228868"])