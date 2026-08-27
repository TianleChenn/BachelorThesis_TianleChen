$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$cohorts = @(
    @{ text="the complete athlete cohort"; filters=@{} },
    @{ text="female competitors"; filters=@{sex="female"} },
    @{ text="male competitors"; filters=@{sex="male"} },
    @{ text="volleyball competitors"; filters=@{sport="volleyball"} },
    @{ text="ice hockey competitors"; filters=@{sport="ice hockey"} },
    @{ text="table tennis competitors"; filters=@{sport="table tennis"} },
    @{ text="3x3 basketball competitors"; filters=@{sport="3x3 basketball"} },
    @{ text="athletes younger than twenty"; filters=@{age_group="under_20"} },
    @{ text="athletes aged twenty or older"; filters=@{age_group="20_and_above"} },
    @{ text="senior national-team members"; filters=@{national_team="Senior"} },
    @{ text="junior national-team members"; filters=@{national_team="Junior"} },
    @{ text="elite competitors"; filters=@{expertise_group="elite"} },
    @{ text="semi-elite competitors"; filters=@{expertise_group="semi_elite"} },
    @{ text="female volleyball competitors"; filters=@{sex="female";sport="volleyball"} },
    @{ text="male ice hockey competitors"; filters=@{sex="male";sport="ice hockey"} },
    @{ text="elite table tennis competitors"; filters=@{expertise_group="elite";sport="table tennis"} },
    @{ text="semi-elite 3x3 basketball competitors"; filters=@{expertise_group="semi_elite";sport="3x3 basketball"} },
    @{ text="female senior national-team members"; filters=@{sex="female";national_team="Senior"} },
    @{ text="male athletes younger than twenty"; filters=@{sex="male";age_group="under_20"} },
    @{ text="elite volleyball athletes aged twenty or older"; filters=@{expertise_group="elite";sport="volleyball";age_group="20_and_above"} }
)
$tasks = @("table1", "table2", "figure1", "correlation", "variance_analysis")
$wording = @{
    table1 = @(
        "Build the predefined Table 1 logistic-regression series for {0}, covering every public domain and the standard no-control, sex, age, and sex-plus-age specifications.",
        "For {0}, return the restricted Table 1 analysis with elite status as target, all public predictors, and each standard control set.",
        "I need Table 1 for {0}: run its four approved logistic models across the eight public athlete domains."
    )
    table2 = @(
        "Build the predefined Table 2 multiple-linear-regression analysis for {0}, using all public domains and expertise value as the continuous outcome.",
        "For {0}, return the restricted Table 2 models for expertise value with the complete public predictor pool.",
        "I need Table 2 for {0}: estimate the approved expertise-value regressions over all eight public domains."
    )
    figure1 = @(
        "Create the approved Figure 1 analysis for {0}, relating all public domains to expertise value and grouping by elite status.",
        "For {0}, generate Figure 1 with the full domain set, the standard correlation threshold, and approved variance iterations.",
        "I need the restricted Figure 1 visualization for {0}, split by elite status and based on every public athlete domain."
    )
    correlation = @(
        "Calculate the approved Pearson correlation analysis across all eight public domains for {0}.",
        "For {0}, return the restricted pairwise Pearson correlation matrix using the complete public domain set.",
        "I need correlations for {0}: use the approved Pearson method over every public athlete domain."
    )
    variance_analysis = @(
        "Compare elite and semi-elite variance across all public domains for {0}, using the approved iteration count and visualization.",
        "For {0}, run the restricted variance analysis by elite status over the complete public domain set.",
        "I need the approved elite-versus-semi-elite variance visualization for {0}, covering every public athlete domain."
    )
}

function New-Dataset([string]$Name, [int]$PerTask, [string]$Prefix, [bool]$Independent) {
    $samples = @()
    $number = 1
    foreach ($task in $tasks) {
        for ($i=0; $i -lt $PerTask; $i++) {
            $cohort = $cohorts[$i]
            $template = $wording[$task][$i % 3]
            $prompt = [string]::Format($template, $cohort.text)
            if ($Independent) { $prompt = "Independent benchmark request: $prompt" }
            $samples += [ordered]@{
                id = "${Prefix}_$($number.ToString('000'))"
                prompt = $prompt
                analysis_type = $task
                requested_filters = $cohort.filters
                difficulty = @("easy", "medium", "hard")[$i % 3]
            }
            $number++
        }
    }
    return [ordered]@{
        dataset_name = $Name
        schema_version = "cloud_local_codegen_v1"
        candidate_samples = $samples.Count
        used_for_training = -not $Independent
        used_for_threshold_calibration = -not $Independent
        independent_evaluation = $Independent
        prompt_version = "pool_typed_schema_v3"
        samples = $samples
    }
}

$training = New-Dataset "athlete_cloud_local_training_prompts_100" 20 "cloud_local_train" $false
$evaluation = New-Dataset "athlete_cloud_local_independent_40" 8 "cloud_local_eval" $true
$normalized = @{}
foreach ($sample in @($training.samples) + @($evaluation.samples)) {
    $key = (($sample.prompt -replace '\s+', ' ').Trim().ToLowerInvariant())
    if ($normalized.ContainsKey($key)) { throw "Duplicate generated prompt: $($sample.prompt)" }
    $normalized[$key] = $true
}

$existingPrompts = @()
Get-ChildItem (Join-Path $Root "evaluation") -Filter *.json | ForEach-Object {
    if ($_.Name -in @("athlete_cloud_local_training_prompts_100.json", "athlete_cloud_local_independent_40.json")) { return }
    try {
        $payload = Get-Content -Raw $_.FullName | ConvertFrom-Json
        foreach ($sample in @($payload.samples)) {
            if ($null -ne $sample.prompt) { $existingPrompts += (($sample.prompt -replace '\s+', ' ').Trim().ToLowerInvariant()) }
        }
    } catch {}
}
foreach ($sample in @($training.samples) + @($evaluation.samples)) {
    $key = (($sample.prompt -replace '\s+', ' ').Trim().ToLowerInvariant())
    if ($existingPrompts -contains $key) { throw "Prompt overlaps an existing evaluation dataset: $($sample.prompt)" }
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $Root "evaluation\athlete_cloud_local_training_prompts_100.json"),
    ($training | ConvertTo-Json -Depth 8), $utf8NoBom)
[IO.File]::WriteAllText((Join-Path $Root "evaluation\athlete_cloud_local_independent_40.json"),
    ($evaluation | ConvertTo-Json -Depth 8), $utf8NoBom)
Write-Host "Generated 100 training and 40 independent evaluation requests with no exact overlap."
