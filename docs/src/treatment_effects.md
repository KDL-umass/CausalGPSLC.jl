# Treatment Effects

## Individual Treatment Effect (ITE)

A contribution of the original GPSLC paper is to produce accurate individual treatment effect conditioned on observed data, using inferred values of latent confounders determined by given structure.

```@docs
sampleITE
```

### Summarizing

It can be helpful to summarize the inferred individual treatment effects into confidence intervals.

```@docs
summarizeITE
```

## Sample Average Treatment Effect (SATE)

Another popular and useful treatment effect estimate is SATE.

```@docs
sampleSATE
```
