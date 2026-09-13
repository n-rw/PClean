using CSV
using DataFrames: DataFrame, nrow

# The CMS "Doctors and Clinicians National Downloadable File" -- the successor to the
# Physician Compare file used in the paper. See PHYSICIANS-DATA.md for provenance and
# for the column mapping from the paper's names to these.
const DATA_PATH = "experiments/physicians_data/DAC_NationalDownloadableFile.csv"

# Columns we model. Everything else in the 31-column file is ignored, as in the paper
# ("Many columns are not modeled").
const COLS = ["NPI", "Cred", "Med_sch", "pri_spec",
              "adr_ln_1", "adr_ln_2", "ZIP Code", "Facility Name", "City/Town"]

row_limit = haskey(ENV, "PHYSICIANS_ROWS") ? parse(Int, ENV["PHYSICIANS_ROWS"]) : nothing

# Force the string columns to String. Left to inference, CSV reads "ZIP Code" as Int
# and silently drops leading zeros (00602 -> 602). NPI stays Int: NumberCodePrior wants one.
const COL_TYPES = Dict("NPI" => Int,
                       "Cred" => String, "Med_sch" => String, "pri_spec" => String,
                       "adr_ln_1" => String, "adr_ln_2" => String, "ZIP Code" => String,
                       "Facility Name" => String, "City/Town" => String)

println("Loading $(DATA_PATH)$(isnothing(row_limit) ? "" : " (first $(row_limit) rows)")...")
dirty_table = CSV.File(DATA_PATH; select=COLS, types=COL_TYPES, stringtype=String,
                       limit=row_limit, ntasks=1) |> DataFrame
println("  $(nrow(dirty_table)) rows")

# Blank strings in a guaranteed (hash-key) field would fragment the index, so coalesce
# missing to "". Cred is deliberately left missing: it is the imputation target.
for c in ["Med_sch", "adr_ln_1", "adr_ln_2", "ZIP Code", "Facility Name", "City/Town"]
    dirty_table[!, c] = map(x -> ismissing(x) ? "" : String(strip(string(x))), dirty_table[!, c])
end
dirty_table[!, "Cred"] = map(x -> ismissing(x) || String(strip(string(x))) == "" ? missing : String(strip(string(x))),
                             dirty_table[!, "Cred"])
dirty_table[!, "pri_spec"] = map(x -> ismissing(x) ? "" : String(strip(string(x))), dirty_table[!, "pri_spec"])

# Blocking key for City, the analogue of the paper's `c2z3`: first two characters of the
# (possibly misspelled) city name plus the first three of the ZIP. Two spellings of the
# same city land in the same block, so the typo model can choose between them, while the
# block stays small enough to enumerate.
dirty_table[!, "c2z3"] = map(eachrow(dirty_table)) do r
    city, zip = r["City/Town"], r["ZIP Code"]
    string(first(city, 2), "_", first(zip, 3))
end

# Candidate city spellings per block -- the `preferring` hint for City.name.
city_candidates = Dict{String, Set{String}}()
for r in eachrow(dirty_table)
    isempty(r["City/Town"]) && continue
    push!(get!(city_candidates, r["c2z3"], Set{String}()), r["City/Town"])
end
const cities = Dict{String, Vector{String}}(k => collect(v) for (k, v) in city_candidates)

const degrees = collect(skipmissing(sort(unique(dirty_table[!, "Cred"]))))
const specialties = sort(unique(filter(!isempty, dirty_table[!, "pri_spec"])))

println("  $(length(degrees)) degrees, $(length(specialties)) specialties, " *
        "$(length(cities)) city blocks (largest $(maximum(length, values(cities))))")
