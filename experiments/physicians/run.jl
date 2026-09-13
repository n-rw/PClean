using PClean

include("load_data.jl")

# The model from the paper, appendix B.4.4, translated to this implementation's surface
# syntax (see README "Differences from the paper"). Two deliberate deviations, both noted
# in PHYSICIANS-DATA.md:
#
#   * `zip` is Unmodeled rather than StringPrior(3, 10). ZIP is 0% blank in the current
#     file, so there is nothing to impute, and a StringPrior over 332k observed ZIPs is
#     pure enumeration cost.
#   * `npi` is @guaranteed. The paper does not index it, but NPI is the clinician
#     identifier and is 0% blank, so indexing it makes physician co-reference an exact
#     lookup instead of a scan over every Physician hypothesized so far. Without this,
#     reference-slot resolution is O(#objects) and millions of rows are hopeless.
PClean.@model PhysiciansModel begin
  @class School begin
    name ~ Unmodeled()
    @guaranteed name
  end;

  @class Physician begin
    @learned error_prob::ProbParameter{1.0, 1000.0}
    @learned degree_dist::Dict{String, ProportionsParameter{3.0}}
    @learned specialty_dist::Dict{String, ProportionsParameter{3.0}}
    npi ~ NumberCodePrior()
    @guaranteed npi
    school ~ School
    begin
      # Degree depends on the school (PCOM awards mostly DOs), specialty on the degree.
      # Both parameter tables are learned from the dirty data.
      degree_probs = degree_dist[school.name]
      degree ~ ChooseProportionally(degrees, degree_probs)
      specialty_probs = specialty_dist[degree]
      specialty ~ ChooseProportionally(specialties, specialty_probs)
      degree_obs ~ MaybeSwap(degree, degrees, error_prob)
    end
  end;

  @class City begin
    c2z3 ~ Unmodeled()
    @guaranteed c2z3
    name ~ StringPrior(3, 30, cities[c2z3])
  end;

  @class Practice begin
    addr ~ Unmodeled()
    @guaranteed addr
    addr2 ~ Unmodeled()
    @guaranteed addr2
    zip ~ Unmodeled()
    @guaranteed zip
    legal_name ~ Unmodeled()
    @guaranteed legal_name
    begin
      # Typos in the city name are modeled at the practice level, so a misspelling
      # repeated across every row of one practice counts as one error, not many votes.
      city ~ City
      city_name ~ AddTypos(city.name, 2)
    end
  end;

  @class Record begin
    physician ~ Physician
    address ~ Practice
  end;
end;

query = @query PhysiciansModel.Record [
  NPI               physician.npi
  Med_sch           physician.school.name
  pri_spec          physician.specialty
  Cred              physician.degree       physician.degree_obs
  adr_ln_1          address.addr
  adr_ln_2          address.addr2
  "ZIP Code"        address.zip
  "Facility Name"   address.legal_name
  c2z3              address.city.c2z3
  "City/Town"       address.city.name      address.city_name
];

config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true, rejuv_frequency=10000)
observations = [ObservedDataset(query, dirty_table)]

@time begin
  trace = initialize_trace(observations, config);
  run_inference!(trace, config);
end

# No ground truth for this dataset, so save the reconstructed latent database instead.
PClean.save_results("results", "physicians", trace, observations)
println("Saved reconstructed database to results/")
