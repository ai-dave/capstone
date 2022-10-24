from . import na_read
import json 
from collections import Counter
from . import LargePrinter as lp


class FASTADataset:
    def __init__(self, dataset_name, fasta_dataset_file, limit=0, lowercase=False):
        self.dataset_name = dataset_name
        self.fasta_dataset_file = fasta_dataset_file
        self.fasta_dataset = na_read.read_fasta(fasta_dataset_file, limit, lowercase)


#    def read_seq(self, accession_number, seq_type):
#        self.fna_file = self.ncbi_dataset[accession_number][seq_type]
#        self.file = "ncbi_dataset/data/" + accession_number + "/" + self.fna_file
#        return na_read.read_fasta_ncbi_dataset_zip(self.ncbi_dataset_zipfile, self.file)
    
#    def get_sequence_report(self, assemblyAccessionNumber):
#        return na_read.make_sequence_report(self.ncbi_dataset_zipfile, assemblyAccessionNumber)


    def kmer_dictionary(self, k, alphabet):
        return na_read.perm(k, alphabet)
    

    def encode_kmer(self, k, alphabet, seq):
        return na_read.make_kmer(k, alphabet, seq)
    

#    def sequence_summary(self, assemblyAccession, seq_type):
#        genes = self.read_seq(assemblyAccession, seq_type)
#        keys_list = list(genes)
#        gene_name = list(genes)[0]
#        gene_sequence = genes[gene_name]
#        print(f'  >[{assemblyAccession}]')
#        print(f'   [{Counter(gene_sequence)}]')
#        print(f'   name: [{gene_name}]')
#        print(f'   sequence: [{gene_sequence[:10]}...{gene_sequence[-10:]}]')
#        print(f'   sequence length: [{len(gene_sequence)}]')
        
#    def kmer_summary(self, assemblyAccession, seq_type, alphabet, k):
#        entry = self.read_seq(assemblyAccession, seq_type)
#        accession_number = list(entry.keys())[0]
#        print(f'>[{accession_number}]')
#        seq = entry[accession_number]
#        print(f'sequence: [{seq[:10]}...{seq[-10:]}]')
#        kmer = self.encode_kmer(k, alphabet, seq)
#        self.print_kmer_vector_reduced(kmer)
        
#    def describe(self, seq_type, base_count_max=16, length_min=0, verbose=0):
#        count_seq_type = 0
#        for accession_number in self.ncbi_dataset:
#            if seq_type in self.ncbi_dataset[accession_number]:
#                entry = self.read_seq(accession_number, seq_type)
#                entry_name = list(entry)[0]
#                sequence = entry[entry_name]
#                if (len(Counter(sequence)) <= base_count_max) and (len(sequence) >=length_min):
#                    count_seq_type += 1
#                if (verbose > 0):
#                    print(f'[{accession_number}] name: [{entry_name}] len:[{len(sequence)}] - [{Counter(sequence)}]')
#
#        print(f'count seq type [{seq_type}]: [{count_seq_type}]')
        
        
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

    def print_kmer_vector_reduced(self, kmer_seq_vector, size=10):
        self.kmer_seq_vector = kmer_seq_vector
        self.size = size
        lpo = lp.LargePrinter()
        lpo.head_tail_list(self.kmer_seq_vector, size=self.size)

    
    def print_kmer_vectors_reduced(self, kmer_vectors, size=10):
        self.size = size
        #lpo = lp.LargePrinter()
        for kmer_vector in kmer_vectors:
            accession_number = list(kmer_vector.keys())[0]
            dataset_name = kmer_vector[accession_number]["dataset_name"]
            kmer_seq_vec = kmer_vector[accession_number]["kmer_seq_vec"]
            print(f'accession_number: [{accession_number}], dataset_name: [{dataset_name}]')
            self.print_kmer_vector_reduced(kmer_seq_vec, size=self.size)

        
