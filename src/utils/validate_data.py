"""Validate the raw Telco churn dataset before preprocessing and training."""

import great_expectations as gx
import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def validate_telco_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate Telco churn schema, allowed values, ranges, and charge consistency.

    Returns:
        A tuple containing the overall validation result and names of failed checks.
    """
    print("🔍 Starting data validation with Great Expectations...")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        failures = [f"column_missing:{column}" for column in missing_columns]
        print(f"❌ Data validation FAILED: missing required columns: {missing_columns}")
        return False, failures

    # The source CSV represents blank TotalCharges cells as strings. Validate a
    # numeric copy so that the raw input remains unchanged for the preprocessing step.
    validation_df = df.copy()
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        validation_df[column] = pd.to_numeric(validation_df[column], errors="coerce")

    # Use GX Core's DataFrame workflow; no on-disk GX project is required.
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("telco_churn_validation")
    data_asset = data_source.add_dataframe_asset(name="telco_customer_churn")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_dataset")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": validation_df})

    expectations = [
        (
            "customerID_not_null",
            gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"),
        ),
        (
            "gender_allowed_values",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="gender", value_set=["Male", "Female"]
            ),
        ),
        (
            "Partner_allowed_values",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="Partner", value_set=["Yes", "No"]
            ),
        ),
        (
            "Dependents_allowed_values",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="Dependents", value_set=["Yes", "No"]
            ),
        ),
        (
            "PhoneService_allowed_values",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="PhoneService", value_set=["Yes", "No"]
            ),
        ),
        (
            "Contract_allowed_values",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="Contract",
                value_set=["Month-to-month", "One year", "Two year"],
            ),
        ),
        (
            "InternetService_allowed_values",
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="InternetService", value_set=["DSL", "Fiber optic", "No"]
            ),
        ),
        (
            "tenure_range",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="tenure", min_value=0, max_value=120
            ),
        ),
        (
            "MonthlyCharges_range",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="MonthlyCharges", min_value=0, max_value=200
            ),
        ),
        (
            "TotalCharges_non_negative",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="TotalCharges", min_value=0
            ),
        ),
        (
            "tenure_not_null",
            gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"),
        ),
        (
            "MonthlyCharges_not_null",
            gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"),
        ),
        (
            "TotalCharges_at_least_MonthlyCharges_for_95_percent",
            gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="TotalCharges",
                column_B="MonthlyCharges",
                or_equal=True,
                mostly=0.95,
            ),
        ),
    ]

    failed_expectations = []
    total_checks = len(expectations)
    for check_name, expectation in expectations:
        result = batch.validate(expectation)
        if not result.success:
            failed_expectations.append(check_name)

    passed_checks = total_checks - len(failed_expectations)
    if failed_expectations:
        print(
            f"❌ Data validation FAILED: {len(failed_expectations)}/"
            f"{total_checks} checks failed"
        )
        print(f"   Failed checks: {failed_expectations}")
    else:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")

    return not failed_expectations, failed_expectations
