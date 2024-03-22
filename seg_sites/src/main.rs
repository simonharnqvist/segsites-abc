use noodles::vcf::Builder;

fn main() -> Result<_, _> {
    let mut reader = Builder::default()
        .build_from_path("../../brenthis/brenthis_data/brenthis_ino_daphne.vcf.gz")?;
    let header = reader.read_header()?;

    for result in reader.records(&header) {
        let record = result;
        println!("{:?}", record);
    }

    Ok()
}
