from . import na_read
import json 
from collections import Counter
from . import LargePrinter as lp


class NCBIDataset:
    def __init__(self, dataset_name, ncbi_dataset_zipfile):
        self.dataset_name = dataset_name
        self.ncbi_dataset_zipfile = ncbi_dataset_zipfile
        self.ncbi_dataset = na_read.make_ncbi_dataset(ncbi_dataset_zipfile)
        self.assembly_data_report = na_read.make_assembly_data_report(self.ncbi_dataset_zipfile)
        
        self.annotationInfo = "" if self.assembly_data_report[0].get("annotationInfo") == None else self.assembly_data_report[0]["annotationInfo"]
        #self.organismName = self.assembly_data_report[0]["organismName"]
        #self.taxId = self.assembly_data_report[0]["taxId"]

        self.assemblyData = {}
        for item in self.assembly_data_report:
            assemblyAccession = item["assemblyInfo"]["assemblyAccession"]
            self.assemblyData[assemblyAccession] = {"assemblyInfo":  item["assemblyInfo"]}
            self.assemblyData[assemblyAccession].update({"assemblyStats": item["assemblyStats"]})
            self.assemblyData[assemblyAccession].update({"organismName": item["organismName"]})
            self.assemblyData[assemblyAccession].update({"taxId": item["taxId"]})


    def files(self, sub=""):
        return list(filter(lambda s: sub in s, na_read.files_ncbi_dataset(self.ncbi_dataset_zipfile)))


    def read_seq(self, accession_number, seq_type):
        self.fna_file = self.ncbi_dataset[accession_number][seq_type]
        self.file = "ncbi_dataset/data/" + accession_number + "/" + self.fna_file
        return na_read.read_fasta_ncbi_dataset_zip(self.ncbi_dataset_zipfile, self.file)
    
    def get_sequence_report(self, assemblyAccessionNumber):
        return na_read.make_sequence_report(self.ncbi_dataset_zipfile, assemblyAccessionNumber)

    def kmer_dictionary(self, k, alphabet):
        return na_read.perm(k, alphabet)
    
    def encode_kmer(self, k, alphabet, seq):
        return na_read.make_kmer(k, alphabet, seq)
    
    def dataset_summary(self):
        # annotation summary
        print(f'dataset name: [{self.dataset_name}]')
        print(f'# fna/fna2 files Total:  [{len(self.files("fna"))}]')
        print(f'# fna cds_from_genomic files: [{len(self.files("cds_from_genomic.fna"))}]')
        print(f'# fna2 genomic files :  [{len(self.files("fna"))-len(self.files("cds_from_genomic.fna"))}]')
        print(f'# gff files:  [{len(self.files("gff"))}]')
        print(f'# faa files:  [{len(self.files("faa"))}]')
        print(json.dumps(self.annotationInfo, indent=2))
        
        accessions = []
        for accession in self.assembly_data_report:
            accessions.append(accession["organismName"])
        lpo = lp.LargePrinter()
        for key, value in Counter(accessions).items():
            lpo.append(f'{key}: {value}')
        #lpo.print()
        lpo.head_tail(size=5)

            
    def assembly_summary(self, assemblyAccession):
        # accession
        print(assemblyAccession)
        print('organismName: [{self.assemblyData[assemblyAccession]["organismName"]}], taxId = [{self.assemblyData[assemblyAccession]["taxId"]}]')
        print('---------------------------------------------------')
        print(json.dumps(self.assemblyData[assemblyAccession]["assemblyInfo"], indent=2))
        print('---------------------------------------------------')
        print(json.dumps(self.assemblyData[assemblyAccession]["assemblyStats"], indent=2))
        if 'fna2' in self.ncbi_dataset[assemblyAccession]:
            print(self.ncbi_dataset[assemblyAccession]['fna2'])
        if 'fna' in self.ncbi_dataset[assemblyAccession]:
            print(self.ncbi_dataset[assemblyAccession]['fna'])
        if 'gff' in self.ncbi_dataset[assemblyAccession]:
            print(self.ncbi_dataset[assemblyAccession]['gff'])
        if 'faa' in self.ncbi_dataset[assemblyAccession]:
            print(self.ncbi_dataset[assemblyAccession]['faa'])

    def sequence_summary(self, assemblyAccession, seq_type):
        genes = self.read_seq(assemblyAccession, seq_type)
        keys_list = list(genes)
        gene_name = list(genes)[0]
        gene_sequence = genes[gene_name]
        print(f'  >[{assemblyAccession}]')
        print(f'   [{Counter(gene_sequence)}]')
        print(f'   name: [{gene_name}]')
        print(f'   sequence: [{gene_sequence[:10]}...{gene_sequence[-10:]}]')
        print(f'   sequence length: [{len(gene_sequence)}]')
        
#    def kmer_summary(self, assemblyAccession, seq_type, alphabet, k):
#        entry = self.read_seq(assemblyAccession, seq_type)
#        accession_number = list(entry.keys())[0]
#        print(f'>[{accession_number}]')
#        seq = entry[accession_number]
#        print(f'sequence: [{seq[:10]}...{seq[-10:]}]')
#        kmer = self.encode_kmer(k, alphabet, seq)
#        self.print_kmer_vector_reduced(kmer)
        
    def describe(self, seq_type, base_count_max=16, length_min=0, verbose=0):
        count_seq_type = 0
        for accession_number in self.ncbi_dataset:
            if seq_type in self.ncbi_dataset[accession_number]:
                entry = self.read_seq(accession_number, seq_type)
                entry_name = list(entry)[0]
                sequence = entry[entry_name]
                if (len(Counter(sequence)) <= base_count_max) and (len(sequence) >=length_min):
                    count_seq_type += 1
                if (verbose > 0):
                    print(f'[{accession_number}] name: [{entry_name}] len:[{len(sequence)}] - [{Counter(sequence)}]')                
        print(f'count seq type [{seq_type}]: [{count_seq_type}]')
        
        
#    def dataset_sequences_as_kmer_vectors(self, seq_type, k, alphabet, base_count_max=16, length_min=0):
#        kmer_vectors = []
#        dict = self.kmer_dictionary(k, alphabet)
#        #count_seq_type = 0
#        for accession_number in self.ncbi_dataset:
#            if seq_type in self.ncbi_dataset[accession_number]:
#                entry = self.read_seq(accession_number, seq_type)
#                entry_name = list(entry)[0]
#                sequence = entry[entry_name]
#                if (len(Counter(sequence)) <= base_count_max) and (len(sequence) >=length_min):
#                    map = {}
#                    map[accession_number] = {"dataset_name": self.dataset_name}
#                    map[accession_number].update({"kmer_seq_vec": self.encode_kmer(k, alphabet, sequence)})
#                    kmer_vectors.append(map)
#        return kmer_vectors

#    def print_kmer_vector_reduced(self, kmer_seq_vector, size=10):
#        self.kmer_seq_vector = kmer_seq_vector
#        self.size = size
#        lpo = lp.LargePrinter()
#        lpo.head_tail_list(self.kmer_seq_vector, size=self.size)

    
#    def print_kmer_vectors_reduced(self, kmer_vectors, size=10):
#        self.size = size
#        #lpo = lp.LargePrinter()
#        for kmer_vector in kmer_vectors:
#            accession_number = list(kmer_vector.keys())[0]
#            dataset_name = kmer_vector[accession_number]["dataset_name"]
#            kmer_seq_vec = kmer_vector[accession_number]["kmer_seq_vec"]
#            print(f'accession_number: [{accession_number}], dataset_name: [{dataset_name}]')
#            self.print_kmer_vector_reduced(kmer_seq_vec, size=self.size)