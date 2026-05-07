import duckdb

con = duckdb.connect("db/lensa.db")

con.sql(
    """
    CREATE TABLE IF NOT EXISTS brreg_data AS
    SELECT
        organisasjonsnummer as id,
        navn as name,
        "organisasjonsform.beskrivelse" as description,
        "naeringskode1.kode" as industry_code,
        "naeringskode1.beskrivelse" as industry_description,
        harRegistrertAntallAnsatte as has_registered_employee_count,
        hjemmeside as website,
        "forretningsadresse.adresse" as address,
        "forretningsadresse.kommune" as city,
        "forretningsadresse.postnummer" as postal_code,
        stiftelsesdato as founding_date,
        sisteInnsendteAarsregnskap as last_submitted_annual_accounts,
        konkurs as bankrupt,
        overordnetEnhet as parent_entity,
        erIKonsern as is_part_of_group,
        aktivitet as activity_description,
        "kapital.belop" as capital_amount
    FROM read_csv('files/enheter-file-march26.csv')
    """
)
