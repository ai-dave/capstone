from . import na_read
from . import NCBIDataset as nds
from . import FASTADataset as fads
from collections import Counter
import numpy as np


class KmerVectors:
    def __init__(self, alphabet, k, ncbidatasets=[], fastadatasets=[], verbose=False):
        self.kmer_vectors = []
        self.alphabet = alphabet
        self.k = k
        self.ncbidatasets = ncbidatasets
        self.fastadatasets = fastadatasets
        self.dict = self.kmer_dictionary(self.k, self.alphabet)
        #self.ncbidataset = ncbidatasets[0]
        #self.ncbi_dataset = ncbidatasets[0].ncbi_dataset
        if verbose:
            print(f'KmerVectors Object -')
            print(f'alphabet [{self.alphabet}]')
            print(f'dict: [{self.dict[:4]}]...[{self.dict[-4:]}]')
        if len(ncbidatasets) > 0:
            self.labels = self.set_Labels_ncbi()
            if verbose:
                print(f'Labels: [{self.labels}]')
                for dataset in ncbidatasets:
                    print(f'[{dataset.dataset_name}]')
                    print(f'[{dataset.ncbi_dataset_zipfile}]')
                    self.ncbi_dataset_summary(dataset)
        elif len(fastadatasets) > 0:
            self.labels = self.set_Labels_fasta()
            if verbose:
                print(f'Labels: [{self.labels}]')
                for dataset in fastadatasets:
                    print(f'[{dataset.dataset_name}]')
                    print(f'[{dataset.fasta_dataset_file}]')

        
    def set_Labels_ncbi(self):
        labels = {}
        i = 1;
        for dataset in self.ncbidatasets:
            labels[dataset.dataset_name]=i
            i += 1
        return labels

    
    def set_Labels_fasta(self):
        labels = {}
        i = 1;
        for dataset in self.fastadatasets:
            labels[dataset.dataset_name]=i
            i += 1
        return labels


    def seqFromFASTA(self, base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        print('seqFromFASTA')
        self.listX = []
        self.listy = []
        for dataset in self.fastadatasets:
            print(f'fasta dataset: [{dataset.dataset_name}]')
            i = 0
            i_using = 0
            skip_count_minlength = 0
            skip_count_alphabet = 0;
            fasta_dataset = dataset.fasta_dataset
            for accession_number in fasta_dataset:
                if dataset_limit > 0 and i_using < dataset_limit:
                    i += 1
                    if verbose:
                        if i % 1000 == 0:
                            print(f'{i}', end = "")
                    if(length_min>0):
                        sequence = fasta_dataset[accession_number][:length_min]
                    else:
                        sequence = fasta_dataset[accession_number]
                    if (len(Counter(sequence)) <= base_count_max) and ((length_min == 0) or (len(sequence) >=length_min)):
                        self.listX.append(sequence)
                        self.listy.append(dataset.dataset_name)
                        i_using += 1
                    else:
                        if len(Counter(sequence)) <= base_count_max:
                            skip_count_alphabet += 1
                        else:
                            skip_count_minlength += 1
                else:
                    if i_using == dataset_limit:
                        if verbose:
                            print(f'capped at [{dataset_limit}]')
                        i_using += 1
            if verbose:
                print(f'-\nTotal:                [{i}]')
                print(f'Using :               [{i_using}]')
                print(f'skip_count_minlength: [{skip_count_minlength}]')
                print(f'skip_count_alphabet: [{skip_count_alphabet}]')
            
        return {'v1': self.listy, 'v2':self.listX}

            

    def seq2KmerSentences(self, seq_type='fna2', base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        if len(self.ncbidatasets) > 0:
            print("NCBI Dataset")
            return(self.seq2KmerSentencesNCBI(seq_type, base_count_max, length_min, dataset_limit, verbose))
        elif len(self.fastadatasets) > 0:
            print("FASTA Dataset")
            return(self.seq2KmerSentencesFASTA(base_count_max, length_min, dataset_limit, verbose))


    def seq2KmerSentencesNCBI(self, seq_type, base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        print('seq2KmerSentencesNCBI')
        self.listX = []
        self.listy = []
        for dataset in self.ncbidatasets:
            print(f'ncbi dataset: [{dataset.dataset_name}]')
            i = 0
            i_using = 0
            skip_count_seqtype = 0
            skip_count_minlength = 0
            skip_count_alphabet = 0;
            ncbi_dataset = dataset.ncbi_dataset
            for accession_number in ncbi_dataset:
                if dataset_limit > 0 and i_using < dataset_limit:
                    i += 1
                    if i % 1000 == 0:
                        print(f'{i}', end = "") 
                    if seq_type in ncbi_dataset[accession_number]:
                        entry = dataset.read_seq(accession_number, seq_type)
                        for sub in entry:
                            #print(key, '->', a_dict[key])
                            #entry_name = entry[sub]
                            sequence = entry[sub][:length_min]
#                            if any(ext in url_string for ext in extensionsToCheck):
#                            if (len(Counter(sequence)) <= base_count_max) and (len(sequence) >=length_min):
                            if (any(ext in sequence for ext in self.dict)) and (len(sequence) >=length_min):
                                self.listX.append(na_read.make_kmer_sentence(self.k, sequence))
                                self.listy.append(dataset.dataset_name)
                                i_using += 1
                            else:
                                if len(Counter(sequence)) <= base_count_max:
                                    skip_count_alphabet += 1
                                else:
                                    skip_count_minlength += 1
                    else:
                        skip_count_seqtype += 1
                else:
                    if i_using == dataset_limit:
                        print(f'capped at [{dataset_limit}]')
                        i_using += 1
            print(f'-\nTotal:                [{i}]')
            print(f'Using :               [{i_using}]')
            print(f'skip_count_seqtype: [{skip_count_seqtype}]')
            print(f'skip_count_minlength: [{skip_count_minlength}]')
            print(f'skip_count_alphabet: [{skip_count_alphabet}]')
            
        return {'v1': self.listy, 'v2':self.listX}

            
    def seq2KmerSentencesFASTA(self, base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        print('seq2KmerSentencesFASTA')
        self.listX = []
        self.listy = []
        for dataset in self.fastadatasets:
            print(f'fasta dataset: [{dataset.dataset_name}]')
            i = 0
            i_using = 0
            skip_count_minlength = 0
            skip_count_alphabet = 0;
            fasta_dataset = dataset.fasta_dataset
            for accession_number in fasta_dataset:
                if dataset_limit > 0 and i_using < dataset_limit:
                    i += 1
                    if verbose:
                        if i % 1000 == 0:
                            print(f'{i}', end = "")
                    if(length_min>0):
                        sequence = fasta_dataset[accession_number][:length_min]
                    else:
                        sequence = fasta_dataset[accession_number]
                    if (len(Counter(sequence)) <= base_count_max) and ((length_min == 0) or (len(sequence) >=length_min)):
                        self.listX.append(na_read.make_kmer_sentence(self.k, sequence))
                        self.listy.append(dataset.dataset_name)
                        i_using += 1
                    else:
                        if len(Counter(sequence)) <= base_count_max:
                            skip_count_alphabet += 1
                        else:
                            skip_count_minlength += 1
                else:
                    if i_using == dataset_limit:
                        if verbose:
                            print(f'capped at [{dataset_limit}]')
                        i_using += 1
            if verbose:
                print(f'-\nTotal:                [{i}]')
                print(f'Using :               [{i_using}]')
                print(f'skip_count_minlength: [{skip_count_minlength}]')
                print(f'skip_count_alphabet: [{skip_count_alphabet}]')
            
        return {'v1': self.listy, 'v2':self.listX}

            
    def seq2KmerVecEncoded(self, seq_type, base_count_max=16, length_min=0):
        #print('seq2KmerVec List')
        self.kmer_vectors = []
        for dataset in self.ncbidatasets:
            print(f'ncbi dataset: [{dataset.dataset_name}]')
            i = 1
            skip_count_seqtype = 0
            skip_count_minlength = 0
            skip_count_alphabet = 0;
            ncbi_dataset = dataset.ncbi_dataset
            for accession_number in ncbi_dataset:
                if i % 100 == 0:
                    print(f'{i}..', end = "") 
                i += 1
                if seq_type in ncbi_dataset[accession_number]:
                    entry = dataset.read_seq(accession_number, seq_type)
                    entry_name = list(entry)[0]
                    sequence = entry[entry_name]
                    if (len(Counter(sequence)) <= base_count_max) and (len(sequence) >=length_min):
                        map = {}
                        map[accession_number] = {"dataset_name": dataset.dataset_name}
                        kmer = self.encode_kmer(sequence)
                        map[accession_number].update({"kmer_seq_vec": kmer})
                        self.kmer_vectors.append(map)
                        #print(f'[{sequence[:4]}] [{kmer[:4]}]')
                    else:
                        if len(Counter(sequence)) <= base_count_max:
                            skip_count_alphabet += 1
                        else:
                            skip_count_minlength += 1
                        #print(' - skipped (alphabet/min-length)')
                else:
                    #print(' - skipped (seq_type)')
                    skip_count_seqtype += 1
            print(f'-\nTotal:                [{i}]')
            print(f'Using :               [{i-skip_count_seqtype-skip_count_minlength-skip_count_alphabet}]')
            print(f'skip_count_seqtype:   [{skip_count_seqtype}]')
            print(f'skip_count_minlength: [{skip_count_minlength}]')
            print(f'skip_count_alphabet:  [{skip_count_alphabet}]')
            print('---------------------------------------------')

            
    def seq2KmerEncodedNumpyVectors(self, seq_type='fna2', base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        if len(self.ncbidatasets) > 0:
            print("NCBI Dataset")
            return(self.seq2KmerEncodedNumpyVectorNCBI(seq_type, base_count_max, length_min, dataset_limit, verbose))
        elif len(self.fastadatasets) > 0:
            print("FASTA Dataset")
            return(self.seq2KmerEncodedNumpyVectorFASTA(base_count_max, length_min, dataset_limit, verbose))


            
    def seq2KmerEncodedNumpyVectorNCBI(self, seq_type, base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        #print('seq2KmerVec Numpy')
        self.listX = []
        self.listy = []
        for dataset in self.ncbidatasets:
            print(f'ncbi dataset: [{dataset.dataset_name}]')
            i = 1
            i_using = 0
            skip_count_seqtype = 0
            skip_count_minlength = 0
            skip_count_alphabet = 0;
            ncbi_dataset = dataset.ncbi_dataset
            for accession_number in ncbi_dataset:
                if dataset_limit > 0 and i_using < dataset_limit:
                    i += 1
                    if i % 1000 == 0:
                        print(f'{i}', end = "") 
                    i += 1
                    if seq_type in ncbi_dataset[accession_number]:
                        entry = dataset.read_seq(accession_number, seq_type)
                        for sub in entry:
                            #print(key, '->', a_dict[key])
                            #entry_name = entry[sub]
                            sequence = entry[sub][:length_min]
    #                        entry_name = list(entry)[0]
    #                        #sequence = entry[entry_name]
    #                        sequence = entry[entry_name][:length_min]
                            if (len(Counter(sequence)) <= base_count_max) and (len(sequence) >=length_min):
                                self.listX.append(self.encode_kmer(sequence))
                                self.listy.append(self.labels[dataset.dataset_name])
                            else:
                                if len(Counter(sequence)) <= base_count_max:
                                    skip_count_alphabet += 1
                                else:
                                    skip_count_minlength += 1
                                #print(' - skipped (alphabet/min-length)')
                    else:
                        #print(' - skipped (seq_type)')
                        skip_count_seqtype += 1
                else:
                    if i_using == dataset_limit:
                        print(f'capped at [{dataset_limit}]')
                        i_using += 1
            print(f'-\nTotal:             [{i}]')
            print(f'Using :               [{i_using}]')
            print(f'skip_count_seqtype:   [{skip_count_seqtype}]')
            print(f'skip_count_minlength: [{skip_count_minlength}]')
            print(f'skip_count_alphabet:  [{skip_count_alphabet}]')
            
        return np.array(self.listX), np.array(self.listy)

                    
    def seq2KmerEncodedNumpyVectorFASTA(self, base_count_max=16, length_min=0, dataset_limit=0, verbose=False):
        print('seq2KmerEncodedNumpyVectorFASTA')
        self.listX = []
        self.listy = []
        for dataset in self.fastadatasets:
            print(f'fasta dataset: [{dataset.dataset_name}], limit: [{dataset_limit}]')
            i = 1
            i_using = 0
            skip_count_minlength = 0
            skip_count_alphabet = 0;
            fasta_dataset = dataset.fasta_dataset
            for accession_number in fasta_dataset:
                if dataset_limit > 0 and i_using < dataset_limit:
                    i += 1
                    if i % 1000 == 0:
                        print(f'{i}', end = "") 
                    sequence = fasta_dataset[accession_number][:length_min].upper()
                    if (len(Counter(sequence)) <= base_count_max) and ((length_min == 0) or (len(sequence) >=length_min)):
                        try:
                            self.listX.append(self.encode_kmer(sequence))
                            self.listy.append(self.labels[dataset.dataset_name])
                            i_using += 1
                            #break
                        except KeyError:
                            print(f'sequence:[{sequence}]')
                    else:
                        if len(Counter(sequence)) <= base_count_max:
                            skip_count_alphabet += 1
                        else:
                            skip_count_minlength += 1
                        #print(' - skipped (alphabet/min-length)')
                else:
                    if i_using == dataset_limit:
                        print(f'capped at [{dataset_limit}]')
                        i_using += 1
            print(f'-\nTotal:             [{i}]')
            print(f'Using :               [{i_using}]')
            print(f'skip_count_minlength: [{skip_count_minlength}]')
            print(f'skip_count_alphabet:  [{skip_count_alphabet}]')
            
        return np.array(self.listX), np.array(self.listy)

                    
    def kmer_dictionary(self, k, alphabet):
        return na_read.perm(k, alphabet)
    
    def encode_kmer(self, seq):
        return na_read.make_kmer(self.k, self.alphabet, seq)
        
    def print_kmer_vector_reduced(self, kmer_seq_vector, size=10):
        self.kmer_seq_vector = kmer_seq_vector
        self.size = size
        print(f'[{self.kmer_seq_vector[:self.size]}]..[{self.kmer_seq_vector[-self.size:]}]')

    
    def print_kmer_vectors_reduced(self, kmer_vectors, size=5):
        self.size = size
        for kmer_vector in kmer_vectors:
            accession_number = list(kmer_vector.keys())[0]
            dataset_name = kmer_vector[accession_number]["dataset_name"]
            kmer_seq_vec = kmer_vector[accession_number]["kmer_seq_vec"]
            print(f'[{dataset_name}] [{accession_number}]', end = "")
            self.print_kmer_vector_reduced(kmer_seq_vec, size=self.size)

    def ncbi_dataset_summary(self, dataset):
        # annotation summary
        print(f'dataset name: [{dataset.dataset_name}]')
        print(f'   fna/fna2 files Total:  [{len(dataset.files("fna"))}]')
        print(f'   fna cds_from_genomic files: [{len(dataset.files("cds_from_genomic.fna"))}]')
        print(f'   fna2 genomic files :  [{len(dataset.files("fna"))-len(dataset.files("cds_from_genomic.fna"))}]')
        print(f'   gff files:  [{len(dataset.files("gff"))}]')
        print(f'   faa files:  [{len(dataset.files("faa"))}]')
        print(f'class labels: [{self.labels}]')

