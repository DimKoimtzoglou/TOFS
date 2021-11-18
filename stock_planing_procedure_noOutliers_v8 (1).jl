
# Pkg.add("NaiveBayes")
# Pkg.add("Plots")

using HDF5,
      DataFrames,
      CSV,
      Dates,
      Statistics,
      StatsBase,
      Distributions,
      Random,
      RollingFunctions,
      GLMNet,
      EvalMetrics,
      Gadfly

import Base.Threads.@spawn

function epochToDate(val_::Int64)::DateTime
      #sec = val ÷ 10^9
      return unix2datetime(val_ ÷ 10^9)
end

function convertToInt32_replaceNaNs(x::Vector{Int64})::Vector{Int32}
      x[findall(k -> isnan(k), x)] .= -1
      return Int32.(x)
end

function convertToFloat32_replaceNaNs(x::Vector{Float64})::Vector{Float32}
      x[findall(k -> isnan(k), x)] .= -1.0
      return Float32.(x)
end

function vectorEpocToDate(x::Vector{Int64})::Vector{DateTime}
      nObs = length(x)
      output = []
      for i in 1:nObs #in eachindex(v_date_sc)
            push!(output, epochToDate(x[i]))
      end
      return output
end

function loadDataActuals(filepath::String)::DataFrame
      data_ = h5read(filepath, "data")

      v_date = @spawn vectorEpocToDate(data_["block3_values"][1, :])
      v_product_id = @spawn convertToInt32_replaceNaNs(data_["block0_values"][1, :])
      v_product_type_id = @spawn convertToInt32_replaceNaNs(data_["block0_values"][2, :])
      v_subsidiary_id = @spawn convertToInt32_replaceNaNs(data_["block0_values"][3, :])
      v_actual_raw = @spawn convertToFloat32_replaceNaNs(data_["block1_values"][1, :])


      output_records = fetch.([v_date, v_product_id, v_product_type_id, v_subsidiary_id, v_actual_raw])
      return DataFrame(output_records, [:dateT, :product_id, :product_type_id, :subsidiary_id, :actual])
end

function loadDataPrice(filepath::String)::DataFrame
      data_ = h5read(filepath, "data")

      v_date = @spawn vectorEpocToDate(data_["block0_values"][1, :])
      v_product_id = @spawn convertToInt32_replaceNaNs(data_["block2_values"][3, :])
      v_product_type_id = @spawn convertToInt32_replaceNaNs(data_["block2_values"][2, :])
      v_subsidiary_id = @spawn convertToInt32_replaceNaNs(data_["block2_values"][1, :])

      v_prices = @spawn convertToFloat32_replaceNaNs(data_["block1_values"][1, :])
      v_lower_tier1 = @spawn convertToFloat32_replaceNaNs(data_["block1_values"][2, :])

      output_records = fetch.([v_date, v_product_id, v_product_type_id, v_subsidiary_id, v_prices, v_lower_tier1])

      return DataFrame(output_records, [:dateT, :product_id, :product_type_id, :subsidiary_id, :price, :price_lowerTier1])
end

function loadData(filepathActuals::String, filepathPrices::String)::DataFrame
      dt_actuals = loadDataActuals(filepathActuals)
      dt_prices = loadDataPrice(filepathPrices)

      output = leftjoin(dt_actuals, dt_prices, on = [:product_id, :product_type_id, :subsidiary_id, :dateT])
      replace!(output.price, missing => 0.0)
      replace!(output.price_lowerTier1, missing => 0.0)

      output[!,:price] = convert.(Float32, output[!,:price])
      output[!,:price_lowerTier1] = convert.(Float32, output[!,:price_lowerTier1])

      return output
end

function loadHistoricalFc(filepath::String)::DataFrame

      data_fc = h5read(filepath, "data")

      v_date = @spawn vectorEpocToDate(data_fc["block2_values"][1, :])
      v_product_id = @spawn convertToInt32_replaceNaNs(data_fc["block0_values"][1, :])
      v_subsidiary_id = @spawn convertToInt32_replaceNaNs(data_fc["block0_values"][2, :])
      v_predictions = @spawn convertToFloat32_replaceNaNs(data_fc["block1_values"][1, :])

      idx_nonans = findall(x -> !isnan(x), data_fc["block1_values"][1, :])

      output_records = fetch.([v_date, v_product_id, v_subsidiary_id, v_predictions])

      return DataFrame(output_records, [:dateT, :product_id, :subsidiary_id, :predictions])[idx_nonans, :]
end

function sumVals(vals)::Int32
      return sum(vals)
end

function findOutlier(vals)::Int32
      nobs = length(vals)

      kend_1 = maximum([1, nobs-1])
      kend_2 = maximum([1, nobs-2])
      kend_3 = maximum([1, nobs-3])
      kend_4 = maximum([1, nobs-4])

      mean_deltas = mean(abs.(diff(vals))) + 1
      delta_k1 = (vals[end] - vals[kend_1]) > (mean_deltas * 3)
      delta_k2 = (vals[end] - vals[kend_2]) > (mean_deltas * 3)
      delta_k3 = (vals[end] - vals[kend_3]) > (mean_deltas * 3)
      delta_k4 = (vals[end] - vals[kend_4]) > (mean_deltas * 3)
      delta_kTot = vals[end] - vals[1] > (mean_deltas * 3)

      if (delta_k1 + delta_k2 + delta_k3 + delta_k4 + delta_kTot) > 1
            output = 1
      else
            output = 0
      end
      return output
end

function splitTrainTest(dtObjective::Union{SubDataFrame, DataFrame}, nlast::Int64)::Tuple{DataFrame, DataFrame}
      if (nrow(dtObjective) - nlast) < 0
            dt_tra = DataFrame()
            dt_tes = DataFrame()
      else
            dt_tra = copy(dtObjective[1:(end-nlast), :])
            dt_tes = copy(dtObjective[(end-nlast+1:end), :])
      end
      return (dt_tra, dt_tes)
end

function createDtwithFeatures(dtObjective::Union{SubDataFrame, DataFrame})::DataFrame

      actuals_ = dtObjective.actual
      prices_ = dtObjective.price
      comprices_ = dtObjective.price_lowerTier1
      idx_comprices_neg = findall(<(0), comprices_)
      comprices_[idx_comprices_neg] .= 0

      if sum(actuals_) > 0
            totObs = nrow(dtObjective)
            initial_val = maximum([findfirst(>(0), actuals_) - 8, 8])

            # feature engineering
            rolln7 = Int32.(running(sumVals, actuals_, 7))
            rollpr7 = runmean(prices_, 7)
            rollcompr7 = runmean(comprices_, 7)

            lagged7_1 = vcat(repeat([0], 1), rolln7[1:(end-1)])
            lagged7_2 = vcat(repeat([0], 2), rolln7[1:(end-2)])
            lagged7_3 = vcat(repeat([0], 3), rolln7[1:(end-3)])
            lagged7_4 = vcat(repeat([0], 4), rolln7[1:(end-4)])
            lagged7_5 = vcat(repeat([0], 5), rolln7[1:(end-5)])
            lagged7_6 = vcat(repeat([0], 6), rolln7[1:(end-6)])
            lagged7_7 = vcat(repeat([0], 7), rolln7[1:(end-7)])

            stepLast21 = repeat([0], totObs)
            stepLast15 = repeat([0], totObs)

            ini21 = maximum([(totObs - 21 + 1), 1])
            ini15 = maximum([(totObs - 15 + 1), 2])

            stepLast21[ini21:end] .= 1
            stepLast15[ini15:end] .= 1

            dt_process = DataFrame([dtObjective.dateT, rollpr7, rollcompr7,
                                    lagged7_1, lagged7_2, lagged7_3, lagged7_4,
                                    lagged7_5, lagged7_6, lagged7_7,
                                    stepLast21, stepLast15, rolln7],
                                   [:dateT, :pr7, :comppr7,
                                   :lagged7_1, :lagged7_2, :lagged7_3, :lagged7_4,
                                   :lagged7_5, :lagged7_6, :lagged7_7,
                                   :stepLast21, :stepLast15, :target7])[initial_val:end, :]

            insertcols!(dt_process, :prod_age => log.(Int32.(1:nrow(dt_process))))

      else
            dt_process = copy(DataFrame())
      end

      return dt_process
end

function formatAndsplitY_X(dtFeatures::DataFrame)::Tuple{Vector{Int32},Matrix{Float32}}
      if nrow(dtFeatures) > 0
            y_ = dtFeatures.target7
            X_ = Matrix(dtFeatures[:, [:pr7, :prod_age, :comppr7,
                                       :lagged7_1, :lagged7_2, :lagged7_3, :lagged7_4,
                                       :lagged7_5, :lagged7_6, :lagged7_7,
                                       :stepLast21, :stepLast15]])
      else
            y_ = [Int32(0)]
            X_ = Array{Float64}(undef, 0, 2)
      end

      return (y_, X_)
end

# Perhaps there is a speed improvement using Static Arrays?
function estimateModel(X_::Matrix{Float32},
                       y_::Vector{Int32},
                       seed_::Int64)::Union{GLMNetCrossValidation, Nothing}

      if size(X_)[1] == 0
            cv_ = nothing
      else
            if length(y_) > 30
                  penalties_ = Float64.([1, 0, 1, 5, 10, 10, 10, 10, 10, 10, 2, 2])
                  constraints_ = Matrix(hcat([-Inf; 0], [-Inf; Inf], [0; Inf],
                                             [-0.9; 0.9], [-0.8; 0.8], [-0.7; 0.7], [-0.6; 0.6],
                                             [-0.5; 0.5], [-0.4; 0.4], [-0.3; 0.3],
                                             [-Inf; Inf], [-Inf; Inf]))

            else
                  penalties_ = Float64.([1, 0, 1, 5, 10, 10, 10, 10, 10, 10, 0, 0])
                  constraints_ = Matrix(hcat([-Inf; 0], [-Inf; Inf], [0; Inf],
                                                   [-0.1; 0.1], [-0.1; 0.1], [-0.1; 0.1], [-0.1; 0.1],
                                                   [-0.1; 0.1], [-0.1; 0.1], [-0.1; 0.1],
                                                   [-Inf; Inf], [-Inf; Inf]))
            end
            obs = size(X_)[1]
            weights_ = repeat([1], obs)
            weights_[(end-2):end] .= 2

            Random.seed!(seed_)
            cv_ = glmnetcv(X_, y_,
                           penalty_factor=penalties_,
                           intercept=true,
                           weights = Float64.(weights_),
                           dfmax=5,
                           constraints=constraints_,
                           alpha=0.8,
                           nfolds=3,
                           parallel=false,
                           tol=1e-8,
                           maxit=8000000)
      end

      return cv_
end

function predictN_ahead(X_::Matrix{Float32},
                        dtForTrain::DataFrame,
                        cvTrained::Union{GLMNetCrossValidation, Nothing},
                        nAhead::Int64)::Tuple{DataFrame, DataFrame, DataFrame}

      if isnothing(cvTrained)
            dt_mod_actuals = DataFrame()
            dt_mod_fitting = DataFrame()
            dt_mod_prediction = DataFrame()
      else
            # fitting
            fitting = GLMNet.predict(cvTrained, X_)
            insertcols!(dtForTrain, :fitting => fitting)

            dt_mod_actuals = dtForTrain[:, [:dateT, :target7]]
            insertcols!(dt_mod_actuals, :class => "actuals")
            rename!(dt_mod_actuals, :target7 => :value)

            dt_mod_fitting = dtForTrain[:, [:dateT, :fitting]]
            insertcols!(dt_mod_fitting, :class => "fitting")
            rename!(dt_mod_fitting, :fitting => :value)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # stochastic daily component
            if sum(dt_mod_actuals.value)>0
                  find_lastSale = findlast(>(0), dt_mod_actuals.value)
            else
                  find_lastSale = 1
            end

            totObs = nrow(dt_mod_actuals)

            iniLastObs = maximum([minimum([find_lastSale, totObs-7]), 1])
            lastObs = dt_mod_actuals.value[iniLastObs:end]
            lastFit = dt_mod_fitting.value[iniLastObs:end]

            x_xhat = lastObs - lastFit

            #Random.seed!(456);
            if (sum(x_xhat) < 0) & (mean(lastObs) < 50)
                  if sum(x_xhat) == 0
                        multiplier = 4
                        stdcomponent = std(x_xhat)/4
                        w1 = 0.9
                        w2 = 0.1
                  else
                        multiplier = 3
                        stdcomponent = std(x_xhat)/3
                        w1 = 0.8
                        w2 = 0.2
                  end
            else
                  multiplier = 0
                  stdcomponent = 0
                  w1 = 0.5
                  w2 = 0.5
            end
            d = Normal((multiplier * mean(x_xhat)), stdcomponent)
            v = rand(d, nAhead)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # initialization
            pr7 = dtForTrain.pr7[end]
            prod_age = log(exp(dtForTrain.prod_age[end]) + 1)
            comppr7 = dtForTrain.comppr7[end]
            lagged7_1 = dtForTrain.target7[end]
            lagged7_2 = dtForTrain.lagged7_1[end]
            lagged7_3 = dtForTrain.lagged7_2[end]
            lagged7_4 = dtForTrain.lagged7_3[end]
            lagged7_5 = dtForTrain.lagged7_4[end]
            lagged7_6 = dtForTrain.lagged7_5[end]
            lagged7_7 = dtForTrain.lagged7_6[end]
            stepLast21 = dtForTrain.stepLast21[end]
            stepLast15 = dtForTrain.stepLast15[end]

            date_obj = dtForTrain.dateT[end] + Dates.Day(1)

            v_dates = []
            v_predictions = []

            for k in 1:nAhead
                  X_1ahead = Matrix(hcat([pr7], [prod_age], [comppr7],
                                   [lagged7_1], [lagged7_2], [lagged7_3], [lagged7_4],
                                   [lagged7_5], [lagged7_6], [lagged7_7],
                                   [stepLast21], [stepLast15]))

                  prediction = w2*maximum([GLMNet.predict(cvTrained, X_1ahead)[1] + v[k], 0.0]) + w1*lagged7_1

                  push!(v_predictions, prediction)
                  push!(v_dates, date_obj)

                  prod_age = log(exp(prod_age) + 1)
                  lagged7_1 = Int32(round(prediction[1], digits=0))
                  lagged7_2 = X_1ahead[1,4]
                  lagged7_3 = X_1ahead[1,5]
                  lagged7_4 = X_1ahead[1,6]
                  lagged7_5 = X_1ahead[1,7]
                  lagged7_6 = X_1ahead[1,8]
                  lagged7_7 = X_1ahead[1,9]
                  date_obj = date_obj + Dates.Day(1)
            end

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # post stochastic component | post adjustment
            iniLastObs = maximum([minimum([find_lastSale, totObs-21]), 1])
            lastObs = dt_mod_actuals.value[iniLastObs:end]
            lastFit = dt_mod_fitting.value[iniLastObs:end]

            x_xhat = lastObs - lastFit

            d = Normal(mean(x_xhat), std(x_xhat))
            v = rand(d, nAhead)

            v2 = repeat([ mean(lastObs[(end-7):end]) ], nAhead)

            v_predictions = v_predictions + v
            v_predictions[findall(<(0), v_predictions)] .= 0
            v_predictions = round.((v_predictions + v2)./2, digits=0)

            if (length(findall(==(0), lastObs)) / length(lastObs)) > 0.9
                  binom_param = length(findall(>(0), lastObs))/minimum([totObs - iniLastObs, 7*12])
                  d2 = Binomial(1, binom_param)
                  v2 = rand(d2, nAhead)

                  idx_binom = findall(>(0), v2)

                  if (length(idx_binom) > 0)
                        for idx in idx_binom
                              if v_predictions[idx] == 0
                                    v_predictions[idx] = 1
                              end
                        end
                  end
            end

            dt_mod_prediction = DataFrame(dateT = v_dates, value = v_predictions)
            dt_mod_prediction[!,:dateT] = convert.(DateTime, dt_mod_prediction[!,:dateT])
            dt_mod_prediction[!,:value] = convert.(Float64, dt_mod_prediction[!,:value])
            insertcols!(dt_mod_prediction, :class => "prediction")

      end

      return (dt_mod_actuals, dt_mod_fitting, dt_mod_prediction)
end

function computePredAndTest(dtInScope::Union{SubDataFrame, DataFrame}, lastDateTrain::Union{DateTime, Nothing})::Tuple{Int32, Int32, Int32, Float64, Float64, String}
      obj_product = dtInScope.product_id[1]
      obj_subsidiary = dtInScope.subsidiary_id[1]

      if !isnothing(lastDateTrain)
            lastDateTrain_plus7 = lastDateTrain + Day(7)
            idx_lastDayTrainTest = findfirst(x -> x == lastDateTrain_plus7, dtInScope.dateT)
            if !isnothing(idx_lastDayTrainTest)
                  dtInScope_trunc = copy(dtInScope[1:idx_lastDayTrainTest, :])
                  toFocusT, toFocusHO = splitTrainTest(dtInScope_trunc, 7)
            else
                  toFocusT = DataFrame()
            end
      else
            toFocusT, toFocusHO = splitTrainTest(dtInScope, 7)
      end

      if nrow(toFocusT) > 15
            last_n = minimum([7, nrow(toFocusT)])
            sum_last = mean(toFocusT.actual[(end-last_n):end]) + 1
            sum_next = mean(toFocusHO.actual) + 1

            if (sum_next - sum_last)/sum_last < -0.50
                  unavailable = "potential lack of availability"
            else
                  unavailable = "--"
            end

            dt_process = createDtwithFeatures(toFocusT)
            y, X = formatAndsplitY_X(dt_process)

            cv = estimateModel(X, y, 123)
            dt_actuals, dt_fitting, dt_prediction = predictN_ahead(X, dt_process, cv, 7)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Identifying sets with not records enough
            if !isnothing(cv)
                  actuals_test = sum(toFocusHO.actual)
                  prediction_ = sum(dt_prediction.value./7)
                  if prediction_ < 1
                        predict_test = ceil(prediction_, digits=0)
                  else
                        predict_test = prediction_
                  end
                  output = (obj_product, obj_subsidiary, actuals_test, predict_test, abs(actuals_test - predict_test), unavailable)
            else
                  output = (obj_product, obj_subsidiary, -1, -1, -1, "lack of records")
            end
      else
            output = (obj_product, obj_subsidiary, -1, -1, -1, "lack of records")
      end
      return output
end

function weighted_mean_absolute_percentage_error(setWithPredAndTest::Vector{Tuple{Int32, Int32, Int32, Float64, Float64, String}})::Float64
      sum_abs_errors = sum([x[5] for x in setWithPredAndTest])
      sum_trues = sum([x[3] for x in setWithPredAndTest]) + 1
      val_metric = (sum_abs_errors / sum_trues)*100
      return val_metric
end

function custom_weighted_mean_absolute_percentage_error(setWithPredAndTest::Vector{Tuple{Int32, Int32, Int32, Float64, Float64, String}})::Float64
      sum_abs_errors = sum([x[5] for x in setWithPredAndTest])
      sum_min_true_pred = sum([minimum([x[3], x[4]]) for x in setWithPredAndTest]) + 1
      val_metric = (sum_abs_errors / sum_min_true_pred)*100
      return val_metric
end

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

start = now()
dt = loadData("./data/stock_plan_actuals_for_assessing.h5", "./data/stock_plan_4year_2month_pricing.h5")
canonicalize(Dates.CompoundPeriod(now() - start))
# ~ 2 minutes

start = now()
dt_fc_prod_ = loadHistoricalFc("./data/stock_plan_fc_capped.h5")
dt_fc_lgbm_ = loadHistoricalFc("./data/stock_plan_fc_lgbm.h5")
canonicalize(Dates.CompoundPeriod(now() - start))

# ------------------------------------------------------------------------------
# just to be sure we do not have repeated values
groupLGBM = groupby(dt_fc_lgbm_, [:dateT, :product_id, :subsidiary_id])
dt_fc_lgbm = combine(groupLGBM, :predictions => mean)
rename!(dt_fc_lgbm, :predictions_mean => :predictions)

groupPROD = groupby(dt_fc_prod_, [:dateT, :product_id, :subsidiary_id])
dt_fc_prod = combine(groupPROD, :predictions => mean)
rename!(dt_fc_prod, :predictions_mean => :predictions)

# GROUP BY :subsidiary_id, :product_id, :dateT
start = now()
DTaggregated = combine(groupby(dt, [:dateT, :product_id, :product_type_id, :subsidiary_id]),
                      [:actual, :price, :price_lowerTier1] =>
                      ((actr, p, plt) -> (
                        actual = sum(actr),
                        price = mean(p),
                        price_lowerTier1 = mean(plt)
                      )) => AsTable)
canonicalize(Dates.CompoundPeriod(now() - start))
# ~ 16 seconds
# ------------------------------------------------------------------------------


# Important -> SORTING VALUES
start = now()
sort!(DTaggregated, [:subsidiary_id, :product_id, :dateT])
canonicalize(Dates.CompoundPeriod(now() - start))
# ~ 1 minute

# aggregating data: just in case for some reason there are not already aggregated
split_dt_ = groupby(DTaggregated, [:subsidiary_id, :product_id])

# filtering the german Subsidiary
idx_NL_BE = []
for i in 1:length(split_dt_)
      subsidiary = split_dt_[i].subsidiary_id[1]
      if subsidiary ≠ 5
            push!(idx_NL_BE, i)
      end
end

split_dt = split_dt_[idx_NL_BE]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# code WARMUP
start = now()
toFocus = split_dt[2342]
warmup_test = computePredAndTest(toFocus, nothing)
canonicalize(Dates.CompoundPeriod(now() - start))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prediction and test in holdout
start = now()
ids_scope = 1:length(split_dt)
output_ = Array{Task}(undef, length(ids_scope))

@inbounds for i in 1:length(ids_scope)
      output_[i] = @spawn computePredAndTest(split_dt[ids_scope[i]], nothing)
      #output_[i] = @spawn computePredAndTest(split_dt[ids_scope[i]], DateTime("2021-06-18"))
      #output_[i] = @spawn computePredAndTest(split_dt[ids_scope[i]], DateTime("2021-03-28"))
end

test_results = fetch.(output_)
canonicalize(Dates.CompoundPeriod(now() - start))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Products with potential lack of availability
potential_lackAvailab = findall(x -> x[6] == "potential lack of availability", test_results)
lack_of_records = findall(x -> x[6] == "lack of records", test_results)

idxWithPotentialAva = setdiff(Set(1:length(split_dt)), union(Set(potential_lackAvailab), Set(lack_of_records)) ) |> collect

WMAE_avail = round(weighted_mean_absolute_percentage_error(test_results[idxWithPotentialAva]), digits=2)
CWMAPE_avail = round(custom_weighted_mean_absolute_percentage_error(test_results[idxWithPotentialAva]), digits=2)
# WMAE_avail = 48.15
# CWMAPE_avail = 64.38


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotting results

actuals_ = [x[3] for x in test_results[idxWithPotentialAva]]
predictions_ = [x[4] for x in test_results[idxWithPotentialAva]]

summarystats(actuals_)
summarystats(predictions_)


dt_actpred = DataFrame([actuals_, predictions_], [:actuals, :predictions])
CSV.write("actual_predictions.csv", dt_actpred)
sum(dt_actpred.actuals)
sum(dt_actpred.predictions)


title_plot = "Actuals vs. Predictions"
ggplot_theme = Theme(panel_fill="gray90", grid_color="white", background_color="white")
set_default_plot_size(15cm, 12cm)
#set_default_plot_size(24cm, 14cm)
abline = Geom.abline(color="red", style=:dash)
p = plot(dt_actpred,
         Coord.cartesian(xmin=0, xmax=800, ymin=0, ymax=800),
         ggplot_theme,
         Guide.title(title_plot),
         Guide.xlabel("actuals"), Guide.ylabel("predictions"),
         layer(x=0:800, y=0:800, Geom.line),
         layer(x="actuals", y="predictions", Geom.hexbin(xbincount=100, ybincount=100)))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# From HERE onwards: COMPARISON with the former results
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

lgbm_date_lastDTrai = dt_fc_lgbm.dateT .- Day.(1)
insertcols!(dt_fc_lgbm, :date_Last => lgbm_date_lastDTrai)
prod_date_lastDTrai = dt_fc_prod.dateT .- Day.(1)
insertcols!(dt_fc_prod, :date_Last => prod_date_lastDTrai)

# ==============================================================================
# ==============================================================================
# COMPARISON to LIGHTGBM

#DateTrain = sort(unique(intersect(Set(unique(dt_fc_prod.date_Last)), Set(unique(dt_fc_lgbm.date_Last))) |> collect))
DateTrain_LGBM = sort(unique(dt_fc_lgbm.date_Last))
ids_scope = 1:length(split_dt)

results_comparison_LGBM = []
results_comparison_LGBM_detailed = []
results_comparison_LGBM_detailed_subs = []
for date_lastT in DateTrain_LGBM

      println(date_lastT)

      dt_fc_lgbm_obj_ = dt_fc_lgbm[dt_fc_lgbm[:,5] .== date_lastT, :]
      dt_fc_lgbm_obj = dt_fc_lgbm_obj_[dt_fc_lgbm_obj_[:,3] .≠ 5, :]

      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # Prediction and test in holdout
      output_ = Array{Task}(undef, length(ids_scope))

      @inbounds for i in 1:length(ids_scope)
            output_[i] = @spawn computePredAndTest(split_dt[ids_scope[i]], date_lastT)
      end

      results_new = fetch.(output_)

      product_ = [x[1] for x in results_new]
      subsidiary_ = [x[2] for x in results_new]
      actuals_ = [x[3] for x in results_new]
      predictionsNew_ = [x[4] for x in results_new]

      dt_results_new_ = DataFrame([product_, subsidiary_, actuals_, predictionsNew_],
                                  [:product_id, :subsidiary_id, :actuals, :predNew])
      dt_results_new = dt_results_new_[dt_results_new_[:,3].≠ -1, :]

      dt_psa_ = DataFrame([product_, subsidiary_, actuals_],[:product_id, :subsidiary_id, :actuals])
      dt_psa = dt_psa_[dt_psa_[:,3].≠ -1, :]


      dt_fc_lgbm_obj_ext = innerjoin(dt_fc_lgbm_obj, dt_psa, on = [:product_id, :subsidiary_id])
      rename!(dt_fc_lgbm_obj_ext, :predictions => :predLGBM)

      dt_comparison_ = innerjoin(dt_fc_lgbm_obj_ext, dt_results_new, on = [:product_id, :subsidiary_id, :actuals])
      push!(results_comparison_LGBM_detailed_subs, (date_lastT, dt_comparison_))

      dt_comparison = combine(groupby(dt_comparison_, [:dateT, :date_Last, :product_id]),
                            [:actuals, :predLGBM, :predNew] =>
                            ((actr, pl, pn) -> (
                              actuals = sum(actr),
                              predLGBM = sum(pl),
                              predNew = sum(pn)
                            )) => AsTable)

      push!(results_comparison_LGBM_detailed, (date_lastT, dt_comparison))

      nProducts = nrow(dt_comparison)

      abERR_predLGBM = abs.(dt_comparison.actuals .- dt_comparison.predLGBM)
      abERR_predNew = abs.(dt_comparison.actuals .- dt_comparison.predNew)

      insertcols!(dt_comparison, :abERR_predLGBM => abERR_predLGBM)
      insertcols!(dt_comparison, :abERR_predNew => abERR_predNew)

      sum_abs_errorsPredLGBM = sum(dt_comparison.abERR_predLGBM)
      sum_abs_errorsPredNew = sum(dt_comparison.abERR_predNew)

      sum_min_true_PredLGBM = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predLGBM))) + 1
      sum_min_true_PredNew = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predNew))) + 1

      sum_trues = sum(dt_comparison.actuals) + 1

      WMAPE_predLGBM = (sum_abs_errorsPredLGBM / sum_trues)*100
      WMAPE_predNew = (sum_abs_errorsPredNew / sum_trues)*100

      CWMAPE_predLGBM = (sum_abs_errorsPredLGBM / sum_min_true_PredLGBM)*100
      CWMAPE_predNew = (sum_abs_errorsPredNew / sum_min_true_PredNew)*100

      push!(results_comparison_LGBM, (date_lastT, nProducts, "LGBM_WMAPE", WMAPE_predLGBM))
      push!(results_comparison_LGBM, (date_lastT, nProducts, "LGBM_CWMAPE", CWMAPE_predLGBM))

      push!(results_comparison_LGBM, (date_lastT, nProducts, "NEW_WMAPE", WMAPE_predNew))
      push!(results_comparison_LGBM, (date_lastT, nProducts, "NEW_CWMAPE", CWMAPE_predNew))
end

outcomeLGBM_dt = DataFrame(results_comparison_LGBM)
rename!(outcomeLGBM_dt, [:dateLTrain, :NProdsInScope, :PredERROR_class, :value])
CSV.write("comparison_LGBM.csv", outcomeLGBM_dt)

detailed_List_dt = [dtt[2] for dtt in results_comparison_LGBM_detailed]
outcomeLGBM_detailed_dt = vcat(detailed_List_dt...)
rename!(outcomeLGBM_detailed_dt, [:date, :dateLTrain, :product_id, :actuals, :predLGBM, :predNew, :abERR_predLGBM, :abERR_predNew])
CSV.write("comparison_detailed_LGBM.csv", outcomeLGBM_detailed_dt)

detailed_List_subs_dt = [dtt[2] for dtt in results_comparison_LGBM_detailed_subs]
outcomeLGBM_detailed_subs_dt = vcat(detailed_List_subs_dt...)
rename!(outcomeLGBM_detailed_subs_dt, [:date, :product_id, :subsidiary_id, :predLGBM, :dateLTrain, :actuals, :predNew])
CSV.write("comparison_detailed_subs_LGBM.csv", outcomeLGBM_detailed_subs_dt)
# ==============================================================================
# ==============================================================================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ==============================================================================
# ==============================================================================
# COMPARISON to PRODUCTION

DateTrain_PROD = sort(unique(dt_fc_prod.date_Last))
ids_scope = 1:length(split_dt)

results_comparison_PROD = []
results_comparison_PROD_detailed = []
results_comparison_PROD_detailed_subs = []

for date_lastT in DateTrain_PROD

      println(date_lastT)

      dt_fc_prod_obj_ = dt_fc_prod[dt_fc_prod[:,5] .== date_lastT, :]
      dt_fc_prod_obj = dt_fc_prod_obj_[dt_fc_prod_obj_[:,3] .≠ 5, :]

      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # Prediction and test in holdout
      output_ = Array{Task}(undef, length(ids_scope))

      @inbounds for i in 1:length(ids_scope)
            output_[i] = @spawn computePredAndTest(split_dt[ids_scope[i]], date_lastT)
      end

      results_new = fetch.(output_)

      product_ = [x[1] for x in results_new]
      subsidiary_ = [x[2] for x in results_new]
      actuals_ = [x[3] for x in results_new]
      predictionsNew_ = [x[4] for x in results_new]

      dt_results_new_ = DataFrame([product_, subsidiary_, actuals_, predictionsNew_],
                                  [:product_id, :subsidiary_id, :actuals, :predNew])
      dt_results_new = dt_results_new_[dt_results_new_[:,3].≠ -1, :]

      dt_psa_ = DataFrame([product_, subsidiary_, actuals_],[:product_id, :subsidiary_id, :actuals])
      dt_psa = dt_psa_[dt_psa_[:,3].≠ -1, :]

      dt_fc_prod_obj_ext = innerjoin(dt_fc_prod_obj, dt_psa, on = [:product_id, :subsidiary_id])
      rename!(dt_fc_prod_obj_ext, :predictions => :predPROD)

      dt_comparison_ = innerjoin(dt_fc_prod_obj_ext, dt_results_new, on = [:product_id, :subsidiary_id, :actuals])
      push!(results_comparison_PROD_detailed_subs, (date_lastT, dt_comparison_))

      dt_comparison = combine(groupby(dt_comparison_, [:dateT, :date_Last, :product_id]),
                            [:actuals, :predPROD, :predNew] =>
                            ((actr, pp, pn) -> (
                              actuals = sum(actr),
                              predPROD = sum(pp),
                              predNew = sum(pn)
                            )) => AsTable)

      push!(results_comparison_PROD_detailed, (date_lastT, dt_comparison))

      nProducts = nrow(dt_comparison)

      abERR_predPROD = abs.(dt_comparison.actuals .- dt_comparison.predPROD)
      abERR_predNew = abs.(dt_comparison.actuals .- dt_comparison.predNew)

      insertcols!(dt_comparison, :abERR_predPROD => abERR_predPROD)
      insertcols!(dt_comparison, :abERR_predNew => abERR_predNew)

      sum_abs_errorsPredPROD = sum(dt_comparison.abERR_predPROD)
      sum_abs_errorsPredNew = sum(dt_comparison.abERR_predNew)

      sum_min_true_PredPROD = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predPROD))) + 1
      sum_min_true_PredNew = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predNew))) + 1

      sum_trues = sum(dt_comparison.actuals) + 1

      WMAPE_predPROD = (sum_abs_errorsPredPROD / sum_trues)*100
      WMAPE_predNew = (sum_abs_errorsPredNew / sum_trues)*100

      CWMAPE_predPROD = (sum_abs_errorsPredPROD / sum_min_true_PredPROD)*100
      CWMAPE_predNew = (sum_abs_errorsPredNew / sum_min_true_PredNew)*100

      push!(results_comparison_PROD, (date_lastT, nProducts, "PROD_WMAPE", WMAPE_predPROD))
      push!(results_comparison_PROD, (date_lastT, nProducts, "PROD_CWMAPE", CWMAPE_predPROD))

      push!(results_comparison_PROD, (date_lastT, nProducts, "NEW_WMAPE", WMAPE_predNew))
      push!(results_comparison_PROD, (date_lastT, nProducts, "NEW_CWMAPE", CWMAPE_predNew))
end

outcomePROD_dt = DataFrame(results_comparison_PROD)
rename!(outcomePROD_dt, [:dateLTrain, :NProdsInScope, :PredERROR_class, :value])
CSV.write("comparison_PROD.csv", outcomePROD_dt)

detailed_List_dt_p = [dtt[2] for dtt in results_comparison_PROD_detailed]
outcomePROD_detailed_dt = vcat(detailed_List_dt_p...)
rename!(outcomePROD_detailed_dt, [:date, :dateLTrain, :product_id, :actuals, :predPROD, :predNew, :abERR_predPROD, :abERR_predNew])
CSV.write("comparison_detailed_PROD.csv", outcomePROD_detailed_dt)

detailed_List_subs_dt_p = [dtt[2] for dtt in results_comparison_PROD_detailed_subs]
outcomePROD_detailed_subs_dt = vcat(detailed_List_subs_dt_p...)
rename!(outcomePROD_detailed_subs_dt, [:date, :product_id, :subsidiary_id, :predPROD, :dateLTrain, :actuals, :predNew])
CSV.write("comparison_detailed_subs_PROD.csv", outcomePROD_detailed_subs_dt)
# ==============================================================================
# ==============================================================================



# HERE ------ TEMPORAL
# iteration over all distinct dates
results_comparison = []
date_lastT = DateTrain[1]


dt_fc_lgbm_obj_ = dt_fc_lgbm[dt_fc_lgbm[:,5] .== date_lastT, :]
dt_fc_lgbm_obj = dt_fc_lgbm_obj_[dt_fc_lgbm_obj_[:,3] .≠ 5, :]
dt_fc_prod_obj_ = dt_fc_prod[dt_fc_prod[:,5] .== date_lastT, :]
dt_fc_prod_obj = dt_fc_prod_obj_[dt_fc_prod_obj_[:,3] .≠ 5, :]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prediction and test in holdout
output_ = Array{Task}(undef, length(ids_scope))

@inbounds for i in 1:length(ids_scope)
      output_[i] = @spawn computePredAndTest(split_dt[ids_scope[i]], date_lastT)
end

results_new = fetch.(output_)

product_ = [x[1] for x in results_new]
subsidiary_ = [x[2] for x in results_new]
actuals_ = [x[3] for x in results_new]
predictionsNew_ = [x[4] for x in results_new]

dt_results_new_ = DataFrame([product_, subsidiary_, actuals_, predictionsNew_],
                            [:product_id, :subsidiary_id, :actuals, :predNew])
dt_results_new = dt_results_new_[dt_results_new_[:,3].≠ -1, :]


dt_psa_ = DataFrame([product_, subsidiary_, actuals_],[:product_id, :subsidiary_id, :actuals])
dt_psa = dt_psa_[dt_psa_[:,3].≠ -1, :]


dt_fc_lgbm_obj_ext = innerjoin(dt_fc_lgbm_obj, dt_psa, on = [:product_id, :subsidiary_id])
rename!(dt_fc_lgbm_obj_ext, :predictions => :predLGBM)

dt_fc_prod_obj_ext = innerjoin(dt_fc_prod_obj, dt_psa, on = [:product_id, :subsidiary_id])
rename!(dt_fc_prod_obj_ext, :predictions => :predPROD)

dt_comparison_ = innerjoin(dt_fc_lgbm_obj_ext, dt_fc_prod_obj_ext, on = [:dateT, :date_Last, :product_id, :subsidiary_id, :actuals])
dt_comparison = innerjoin(dt_comparison_, dt_results_new, on = [:product_id, :subsidiary_id, :actuals])

nProducts = nrow(dt_comparison)

abERR_predLGBM = abs.(dt_comparison.actuals .- max.(dt_comparison.predLGBM, 0))
abERR_predPROD = abs.(dt_comparison.actuals .- max.(dt_comparison.predPROD, 0))
abERR_predNew = abs.(dt_comparison.actuals .- max.(dt_comparison.predNew, 0))

insertcols!(dt_comparison, :abERR_predLGBM => abERR_predLGBM)
insertcols!(dt_comparison, :abERR_predPROD => abERR_predPROD)
insertcols!(dt_comparison, :abERR_predNew => abERR_predNew)

sum_abs_errorsPredLGBM = sum(dt_comparison.abERR_predLGBM)
sum_abs_errorsPredPROD = sum(dt_comparison.abERR_predPROD)
sum_abs_errorsPredNew = sum(dt_comparison.abERR_predNew)

sum_min_true_PredLGBM = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predLGBM))) + 1
sum_min_true_PredPROD = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predPROD))) + 1
sum_min_true_PredNew = sum(min.(abs.(dt_comparison.actuals), abs.(dt_comparison.predNew))) + 1

sum_trues = sum(dt_comparison.actuals) + 1

WMAPE_predLGBM = (sum_abs_errorsPredLGBM / sum_trues)*100
WMAPE_predPROD = (sum_abs_errorsPredPROD / sum_trues)*100
WMAPE_predNew = (sum_abs_errorsPredNew / sum_trues)*100

CWMAPE_predLGBM = (sum_abs_errorsPredLGBM / sum_min_true_PredLGBM)*100
CWMAPE_predPROD = (sum_abs_errorsPredPROD / sum_min_true_PredPROD)*100
CWMAPE_predNew = (sum_abs_errorsPredNew / sum_min_true_PredNew)*100

push!(results_comparison, (date_lastT, nProducts, "LGBM_WMAPE", WMAPE_predLGBM))
push!(results_comparison, (date_lastT, nProducts, "LGBM_CWMAPE", CWMAPE_predLGBM))

push!(results_comparison, (date_lastT, nProducts, "PROD_WMAPE", WMAPE_predPROD))
push!(results_comparison, (date_lastT, nProducts, "PROD_CWMAPE", CWMAPE_predPROD))

push!(results_comparison, (date_lastT, nProducts, "NEW_WMAPE", WMAPE_predNew))
push!(results_comparison, (date_lastT, nProducts, "NEW_CWMAPE", CWMAPE_predNew))
