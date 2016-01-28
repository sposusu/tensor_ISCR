import pdb

# Define Cost Table
def genCostTable():
  values = [ -30., -10., -50., -20., 0., 0., 1000. ]
  costTable = dict(zip(range(6)+['lambda'],values))
  return costTable
