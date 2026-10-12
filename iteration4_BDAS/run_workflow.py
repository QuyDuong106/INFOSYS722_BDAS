"""Primary Iteration 4 entrypoint: native PySpark preparation and holdout CV."""
from pathlib import Path
import sys,runpy
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    spark_dir=root if (root/'spark_iterations.py').exists() else root.parent/'PySpark'
    sys.path.insert(0,str(spark_dir))
    runpy.run_module('spark_iterations',run_name='__main__')
