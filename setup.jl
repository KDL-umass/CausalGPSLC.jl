using Pkg

Pkg.activate("GPSLCenv")

Pkg.add(PackageSpec(url="https://github.com/probcomp/Gen"))

Pkg.add("Random")
Pkg.add("CSV")
Pkg.add("ArgParse")
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("ProgressBars")
Pkg.add("Statistics")
Pkg.add("Distributions")
Pkg.add("FunctionalCollections")