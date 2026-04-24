import argparse
import subprocess
import gffutils
from Bio import SeqIO
import os
import sys
import pandas as pd
import yaml

from gffutils.exceptions import FeatureNotFoundError


# Helper function to get a dict which saves for every transcript which bases are the Start of a Codon
def get_cds_phase_map(db, transcript_id):
    # We get all bases in the CDS ...
    cds_features = list(db.children(transcript_id, featuretype='CDS', order_by='start'))
    # ... as well as the STOP Codon
    stop_features = list(db.children(transcript_id, featuretype='stop_codon', order_by='start'))
    coding_markers = cds_features + stop_features
    # This dict saves, whether the position is the Start of a Codon
    codon_start_map = {}

    for cds in coding_markers:
        # We use the frame of the transcript
        try:
            frame = int(cds.frame)
        except (ValueError, TypeError):
            frame = 0  # Fallback, if '.'

        # Ensembl GTF Phase Definition:
        # Phase 0: The first Codon starts at the first base (the start) of the feature
        # Phase 1: The first Codon starts at the second base of the feature
        # Phase 2: The first Codon starts at the third base of the feature

        if cds.strand == '+':
            start_pos = cds.start
            end_pos = cds.end
            # We iterate over a CDS
            # current_base_index is 0-based relative to the start of the feature
            for current_base_index in range(end_pos - start_pos + 1):
                genomic_pos = start_pos + current_base_index

                # Consider the start of the feature where current_base_index = 0
                # If frame = 0 --> current_base_index - frame = 0 --> 0 % 3 = 0 --> The first Base (which starts at positon 0) is the start of a Codon
                # If frame = 1 --> current_base_index - frame = -1 --> -1 % 3 = 2 --> The fist Base is NOT the start of a codon
                # If frame = 2 --> current_base_index - frame = -2 --> -2 % 3 = 1 --> The fist Base is NOT the start of a codon
                if (current_base_index - frame) % 3 == 0 and (current_base_index - frame) >= 0:
                    codon_start_map[genomic_pos] = True
                else:
                    codon_start_map[genomic_pos] = False

        else:
            # Minus Strand Logic
            # The 5' end is at the minus strand at the genomic "end"
            # Frame refers to the 5' end
            start_pos = cds.start # 3' end
            end_pos = cds.end # 5' end

            for current_base_index in range(end_pos - start_pos + 1):
                # At a minus strand the sequence begins at the end --> We just have to adjust the genomic position which now starts at end_pos and goes to start_pos
                genomic_pos = end_pos - current_base_index

                if (current_base_index - frame) % 3 == 0 and (current_base_index - frame) >= 0:
                    codon_start_map[genomic_pos] = True
                else:
                    codon_start_map[genomic_pos] = False
    return codon_start_map


def create_gtf_db(gtf_file, db_path, force=True):
    # Only needs to be executed once
    print(f"Creating GFF database at {db_path} (this may take a while)...", flush=True)
    db = gffutils.create_db(
        gtf_file,
        dbfn=db_path,
        force=True,
        keep_order=True,
        merge_strategy='merge',
        sort_attribute_values=True,
        disable_infer_genes=True,
        disable_infer_transcripts=True
    )
    print("Database created.", flush=True)
    return db


def run_gffread(genome_fasta, annotation_gtf, output_transcripts):
    print(f"Extracting transcripts using gffread to {output_transcripts}...", flush=True)
    
    # Check if we are running in a conda environment or have access to one
    # Note: Using bash shell block here to handle conda activation safely
    script_content = f"""
    # Try to initialize conda for the bash session
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        . ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        . ~/anaconda3/etc/profile.d/conda.sh
    else
        # Fallback to pure environment activation if conda is in path
        eval "$(conda shell.bash hook 2> /dev/null)"
    fi
    
    # Check if the environment exists
    if ! conda env list | grep -q 'bio-tools'; then
        echo "ERROR: Conda environment 'bio-tools' not found."
        echo "Please create it and install gffread:"
        echo "  conda create -n bio-tools -c bioconda gffread"
        echo "  conda activate bio-tools"
        exit 1
    fi

    conda activate bio-tools
    
    # Check if gffread is actually available
    if ! command -v gffread &> /dev/null; then
        echo "ERROR: gffread could not be found even inside the 'bio-tools' environment."
        echo "Please install it:"
        echo "  conda activate bio-tools"
        echo "  conda install -c bioconda gffread"
        exit 1
    fi
    
    gffread "{annotation_gtf}" -g "{genome_fasta}" -w "{output_transcripts}"
    """
    
    with open("temp_gffread.sh", "w") as f:
        f.write(script_content)
        
    try:
        # Run using bash to fully interpret the environment activation
        result = subprocess.run(["bash", "temp_gffread.sh"], check=False, text=True, capture_output=True)
        
        if result.returncode != 0:
            print("Failed to execute gffread script!", flush=True)
            print("STDOUT:", result.stdout, flush=True)
            print("STDERR:", result.stderr, flush=True)
            sys.exit(1)
            
        print("Transcripts successfully extracted.", flush=True)
    finally:
        if os.path.exists("temp_gffread.sh"):
            os.remove("temp_gffread.sh")


def process_transcripts(db_path, transcript_fasta_file, output_file, ej_markers=True, cds_markers=True):
    print(f"Loading GTF database: {db_path}", flush=True)
    db = gffutils.FeatureDB(db_path)

    print(f"FASTA-file: {transcript_fasta_file}", flush=True)
    print(f"Output file: {output_file}", flush=True)
    try:
        with open(output_file, "w") as outfile:

            # These variables can be derived from the GTF file
            header_cols = [
                "ensembl_transcript_id",
                "ensembl_gene_id",
                "hgnc_symbol",
                "transcript_biotype",
                "sequence"
            ]
            outfile.write("\t".join(header_cols) + "\n")

            # We iterate over every entry in the FASTA file with all transcripts
            # --> This file was generated using gffread, which processes a FASTA file containing the genome and a corresponding GTF file
            record_count = 0
            processed_count = 0

            with open(transcript_fasta_file, "rt") as fasta_handle:
                for record in SeqIO.parse(fasta_handle, "fasta"):

                    record_count += 1

                    # Normalize the IDs in case they have versioning like ENSG00000227232.2
                    transcript_id_base = record.id.split('.')[0]
                    sequence_str = str(record.seq)
                    try:
                        # We get the additional features for the transcript from the GTF file db
                        feature = db[transcript_id_base]

                        gene_id = feature.attributes.get('gene_id', ['N/A'])[0]
                        gene_name = feature.attributes.get('gene_name', ['N/A'])[0]  # hgnc_symbol
                        biotype = feature.attributes.get('transcript_biotype', ['N/A'])[0]

                        # We get all exons from the GTF db
                        exons = list(db.children(feature, featuretype='exon', order_by='start'))

                        if not exons:
                            sys.stderr.write(f"WARN: No exons found for {transcript_id_base}. Skipping.\n")
                            continue  # Next transcript

                        is_negative_strand = (feature.strand == '-')
                        # The FASTA File always returns 5' --> 3'. If the strand is negative, we want to reverse that
                        if is_negative_strand:
                            exons.reverse()

                        exon_lengths = [len(e) for e in exons]

                        # The length of the exons defined in the GTF has to match the length of the transcript:
                        # If it doesn't match, something is wrong, could be a mismatch between the original GTF File and the genome FASTA file
                        if sum(exon_lengths) != len(sequence_str):
                            sys.stderr.write(f"WARN: Length missmatch for {transcript_id_base}. "
                                             f"Seq: {len(sequence_str)}, Exons: {sum(exon_lengths)}. Skipping.\n")
                            continue

                        # --- Determine CDS boundaries ---
                        # We create a codon map, which tells us for every position if the base is at the start of a codon
                        codon_start_map = get_cds_phase_map(db, transcript_id_base)
                        # We get the CDS and the stop codon from the gtf file
                        cds_features = list(db.children(feature, featuretype='CDS'))
                        stop_features = list(db.children(feature, featuretype='stop_codon'))

                        # We combine both --> the stop codon will also be marked as the last codon with a capital base
                        coding_markers = cds_features + stop_features

                        genomic_cds_start = None
                        genomic_cds_end = None

                        if coding_markers:
                            genomic_cds_start = min(c.start for c in coding_markers)
                            genomic_cds_end = max(c.end for c in coding_markers)

                        # Format the sequence for saluki
                        exon_sequences = []
                        current_pos = 0

                        # We iterate over every exon length provided by the GTF file
                        for i, exon in enumerate(exons):
                            raw_exon_seq = sequence_str[current_pos: current_pos + len(exon)]
                            formatted_chars = []

                            # We have to genomically iterate over the exons
                            if not is_negative_strand:
                                range_iter = range(exon.start, exon.end + 1)  # Genomic coordinates
                            else:
                                # Minus strand: We have reversed the sequence --> we also need to iterate reversed over the sequence to match the codon dict we have created
                                range_iter = range(exon.end, exon.start - 1, -1)

                            for char_idx, genomic_pos in enumerate(range_iter):
                                char = raw_exon_seq[char_idx]

                                if not cds_markers:
                                    formatted_chars.append(char.lower())
                                else:
                                    # Check if the base is in a codon
                                    if genomic_pos in codon_start_map:
                                        # Check if the base is the start of a codon
                                        is_start = codon_start_map[genomic_pos]
                                        if is_start:
                                            formatted_chars.append(char.upper())
                                        else:
                                            formatted_chars.append(char.lower())
                                    else:
                                        # UTR
                                        formatted_chars.append(char.lower())

                            exon_sequences.append("".join(formatted_chars))
                            current_pos += len(exon)

                        # Add the junction markers
                        final_seq_parts = []
                        for i, exon_seq in enumerate(exon_sequences):
                            if i == 0 or not ej_markers:
                                # The first base is not marked as a junction
                                final_seq_parts.append(",".join(list(exon_seq)))
                            else:
                                first_base = exon_seq[0]  # The marker "ej" is adjusted to the base
                                if first_base.isupper():
                                    # If the base was the start of a codon --> markes also capital
                                    marker_type = first_base + "EJ"  # e.g. AEJ
                                else:
                                    # If the base was not the start of a codon --> markes also lower case
                                    marker_type = first_base + "ej"  # e.g. aej

                                final_seq_parts.append(marker_type)

                                rest = exon_seq[1:]
                                if rest:
                                    final_seq_parts.append(",".join(list(rest)))

                        final_sequence_string = ",".join(final_seq_parts)

                        output_data = [
                            transcript_id_base,
                            gene_id,
                            gene_name,  # hgnc_symbol
                            biotype,
                            final_sequence_string
                        ]

                        outfile.write("\t".join(output_data) + "\n")
                        processed_count += 1

                    except FeatureNotFoundError:
                        sys.stderr.write(f"WARN: ID {transcript_id_base} not found in GTF-DB. Skipping.\n")
                        continue
                    except Exception as e:
                        sys.stderr.write(f"ERROR when processing {transcript_id_base}: {e}. Skipping.\n")
                        continue
                    if record_count % 1000 == 0:
                        print(f"  ... {record_count} transcripts processed.", flush=True)

    except FileNotFoundError:
        print(f"ERROR: The FASTA file '{transcript_fasta_file}' was not found.", file=sys.stderr, flush=True)
        exit(1)
    except IOError as e:
        print(f"ERROR when writing the file '{output_file}': {e}", file=sys.stderr, flush=True)
        exit(1)

    print("\n--- Processing Finished ---", flush=True)
    print(f"Record count in FASTA file: {record_count}", flush=True)
    print(f"Successfully processed: {processed_count}", flush=True)
    print(f"Result saved in: {output_file}", flush=True)


def merge_csv_data(txt_file, yaml_config_path):
    print(f"\nLoading merge configuration from {yaml_config_path}...", flush=True)
    with open(yaml_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    csv_path = config.get('csv_path')
    columns_to_append = config.get('columns_to_append', [])
    txt_id_col = config.get('txt_id_col')
    csv_id_col = config.get('csv_id_col')

    if not all([csv_path, txt_id_col, csv_id_col]):
        print("ERROR: YAML config missing required fields (csv_path, txt_id_col, csv_id_col).", flush=True)
        return

    print("Reading text and CSV files...", flush=True)
    df_txt = pd.read_csv(txt_file, sep='\t')
    df_csv = pd.read_csv(csv_path)

    # 1-based index mapped to 0-based for extraction
    try:
        txt_id_col_name = df_txt.columns[int(txt_id_col) - 1]
    except IndexError:
        print(f"ERROR: TXT ID column index {txt_id_col} is out of bounds.", flush=True)
        return

    try:
        csv_id_col_name = df_csv.columns[int(csv_id_col) - 1]
    except IndexError:
        print(f"ERROR: CSV ID column index {csv_id_col} is out of bounds.", flush=True)
        return

    # Prepare columns to keep from the CSV file
    cols_to_keep = [csv_id_col_name]
    for col in columns_to_append:
        if isinstance(col, int):
            # 1-based index handling for columns to append
            try:
                cols_to_keep.append(df_csv.columns[col - 1])
            except IndexError:
                print(f"ERROR: Column index {col} to append is out of bounds in CSV.", flush=True)
        else:
            if col in df_csv.columns:
                cols_to_keep.append(col)
            else:
                print(f"WARN: Column '{col}' not found in CSV.", flush=True)
    
    # Remove duplicates to avoid merge errors
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    df_csv_subset = df_csv[cols_to_keep]

    print(f"Merging on TXT column '{txt_id_col_name}' and CSV column '{csv_id_col_name}'...", flush=True)
    
    initial_length = len(df_txt)
    df_merged = df_txt.merge(df_csv_subset, left_on=txt_id_col_name, right_on=csv_id_col_name, how='left')

    if txt_id_col_name != csv_id_col_name:
        df_merged.drop(columns=[csv_id_col_name], inplace=True)
        
    csv_keys = set(df_csv[csv_id_col_name].dropna().unique())
    merged_count = df_txt[txt_id_col_name].isin(csv_keys).sum()
    unmerged_count = initial_length - merged_count

    print(f"\n--- Merging Finished ---", flush=True)
    print(f"Rows successfully merged (from the original TXT): {merged_count}", flush=True)
    print(f"Rows not found in CSV (not merged): {unmerged_count}", flush=True)

    if len(df_merged) > initial_length:
        print(f"WARN: Due to duplicates in the CSV keys, {initial_length} rows became {len(df_merged)} rows after the merge.", flush=True)

    print("Saving updated file...", flush=True)
    df_merged.to_csv(txt_file, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description="Single-command Preprocessing Pipeline for extracting and enriching transcripts.")
    parser.add_argument("genome_fasta", help="Path to the toplevel FASTA file (e.g. Homo_sapiens.GRCh38.dna.toplevel.fa)")
    parser.add_argument("annotation_gtf", help="Path to the corresponding GTF file (e.g. sorted.gtf)")
    parser.add_argument("output_dir", help="Directory where all output files (DB, extracted FASTA, final TXT) will be stored")
    parser.add_argument("--create_db", action="store_true", help="Forces the creation of a new GTF SQLite database. If omitted, uses existing DB if available.")
    parser.add_argument("--output_txt", default="enriched_transcripts.txt", help="Name of the final enriched transcripts output file (default: enriched_transcripts.txt)")
    parser.add_argument("--ej_markers", type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True, help="Add exon junction markers (default: True)")
    parser.add_argument("--cds_markers", type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True, help="Add CDS markers (upper/lower case) (default: True)")
    parser.add_argument("--merge_yaml", help="Path to a YAML configuration file for merging additional CSV columns into the output.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define paths for intermediate & output files dynamically based on input GTF name
    gtf_basename = os.path.basename(args.annotation_gtf)
    db_path = os.path.join(args.output_dir, f"{gtf_basename}.db")
    extracted_fa_path = os.path.join(args.output_dir, "extracted_transcripts.fa")
    final_output_path = os.path.join(args.output_dir, args.output_txt)

    # Step 1: Run gffread if the transcripts are not yet extracted
    if not os.path.exists(extracted_fa_path):
        run_gffread(args.genome_fasta, args.annotation_gtf, extracted_fa_path)
    else:
        print(f"File {extracted_fa_path} already exists. Skipping gffread extraction.", flush=True)

    # Step 2: Create GTF DB if it doesn't exist or if forced
    if args.create_db or not os.path.exists(db_path):
        create_gtf_db(args.annotation_gtf, db_path, force=True)
    else:
        print(f"Using existing GTF database: {db_path}", flush=True)

    # Step 3: Run the enrichment process
    process_transcripts(db_path, extracted_fa_path, final_output_path, ej_markers=args.ej_markers, cds_markers=args.cds_markers)

    # Step 4: Merge CSV data if YAML is provided
    if args.merge_yaml:
        if os.path.exists(args.merge_yaml):
            merge_csv_data(final_output_path, args.merge_yaml)
        else:
            print(f"ERROR: YAML configuration file '{args.merge_yaml}' not found.", flush=True)


if __name__ == "__main__":
    main()