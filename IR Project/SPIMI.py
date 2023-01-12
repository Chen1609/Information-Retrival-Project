
def SPIMI_INVERT(tokens):
    # output_file = new_file
    dictionary = {}
    while(free memory available):
        for token in tokens:
            if token.term not in dictionary:
                posting_list = ADDTODICTIONARY(dictionary, term.token)
            else:
                posting_list = GETPOSTINGLIST(dictionary, term.token)
            if full(posting_list):
                posting_list = DOUBLEPOSTINGLIST(dictionary, term.token)
            ADDTOPOSTINGLIST(posting_list, token.doc_id)
    sorted_terms = SORTTERMS(dictionary)
    WRITEBLOCKTODISK(sorted_terms, dictionary, outputfile)
    return output_file

