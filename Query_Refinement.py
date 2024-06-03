
import csv
import json
class Query_Refinement:
    def refine_query(self, query):
        import nltk
        from nltk.corpus import wordnet

        # tokenize the query
        query_tokens = nltk.word_tokenize(query)

        # create a set to store the refined query
        refined_query = set(query_tokens)

        # loop through each token and add related words
        for token in query_tokens:
            # find synonyms and related words for the token
            synsets = wordnet.synsets(token)
            for synset in synsets:
                for lemma in synset.lemmas():
                    # add the lemma to the refined query if it's not the same as the original token
                    if lemma.name() != token:
                        refined_query.add(lemma.name())

        # return the refined query as a string
        return " ".join(refined_query)

    def refine_queries_file(self, input_file, output_file):
       
         # Open the input and output files
        with open(input_file, 'r', newline='') as f_in, open(output_file, 'w', newline='') as f_out:
            # Initialize CSV writer
            writer = csv.writer(f_out)

            # Read and process each line in the JSONL file
            for line in f_in:
                # Parse JSON object from the current line
                item = json.loads(line)
                
                id_left = item['qid']
                text_left = item['query']
                
                # Apply query refinement to the text column (query)
                refined_query = self.refine_query(text_left)
                
                # Write the ID and refined query to the output file
                writer.writerow([id_left, refined_query])

        # Return the output file path
        return output_file