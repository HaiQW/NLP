# The MIT Liscense
# Copyright(c) 2015 Thoughtly, Corp
#
# Code: url = https://github.com/pthomas1/MLBlog/blob/master/words.py
# Toturial url = http://www.thoughtly.co/blog/working-with-text/
#
# Description: Lesson 4: Naive bayes classifier
#
#
#
#
# Argparse is a standard python mechanism for handling commandline args while
# avoiding a buncho of boilerplate code.
import argparse

# This is a module that provies a bunch of simple methods that make accessing
# the filesystem simpler
from utils import fs, log

import math

# python logging allows us to log formatted log messages at different log levels
import logging

# pulls in tokenizing and import of corpus
import words

# Used for stopwords
import nltk

# used for float min
import sys

# used for read in unicode files
import codecs


################################################################################
#
# We can choose to do one or both training and classifying of a document using a
# Naive Bayes Classifier.
#
################################################################################

def main():

    # Build the commandline parser and return args.  This also setups up any non
    # ML/NLP config needed by the script(such as logging)
    args = configure_command_line_arguments()

    # If we are traing the classifier
    if args["train"]:
        train_classifier(args)

    # If we are classifying
    if args["classify"] is not None:
        class_name,log_probability = classify(args)
        logging.info("Classified into " + class_name + " with probability " + str(log_probability))




################################################################################
#
# Classify a document using the Naive Bayes Classifier.  We make one
# Simplification in this classifie - we are tracking data to the class level and
# not the document level( because some of the corpora don't have that info).
# This means that our priors for the classifier are all 1/number of class.  This
# doesn't make a large difference in math behind the classifier.  It is equivale
# nt to a set of calsses with identical numbers of documents(and here we are sy
# ing each class has precisely one document)
#
################################################################################

def classify(args):
    file_name =  args["classify"]
    logging.debug("Classify " + file_name)

    # Load the trainging data and class names.
    training_data, class_names = load_training_data()

    # Read from the document to classify
    to_classify = codecs.open(args["classify"],"r","utf-8").read()

    # Tokenize the document to classify.
    to_classify_terms = nltk.word_tokenize(to_classify)

    # If we have enabled stemming then stem these words
    if args["stemming"]:
        to_classify_terms = words.stem_words_array(to_classify_terms)

    # We are now ready to actually classify the document.  We need to determine
    # the probability that our document(D) is a member of each of our class(C).
    # We calculate this probability by taking the product of the probability tha
    # t each word in the document belongs to the class c (this is the naive
    # aspect of all other word probabilities).  This calculates the probability
    # of the words in this documents given a class c - > p(w|c)
    class_probabilities = {}

    # In this example, each class is comprised of just one document.  The
    # probability that a document falls in a class is therefore 1/the number of
    # classes.  We use the log probability to counteract the effect of the
    # product of many near-0 probabilities.  In our example we are actually
    # calling each corpus a single document, so the probability of a given
    # documents would have higher probabilities of being picked by the classifi
    # er because this term would be relative high when compared to other
    # categories.
    log_probability_of_class = math.log(1.0 / len(class_names))

    stopwords = nltk.corpus.stopwords.words('english')

    # We need the total vocabulary size in order to do laplace smoothing
    vocabulary_size = calculate_vocabulary_size(training_data)
    logging.debug("Total_vocabulary_size "+ str(vocabulary_size) + " terms")

    # Calculate the word probability product for each class p(w|c)
    for class_name in class_names:
        logging.debug("Calculating log probability for class "+ class_name)

        # Keep everything log probabilities -math.log(1) = 0
        log_probability_of_words_in_class = math.log(1)

        # We need the number of terms in the class
        number_of_terms_in_class = calculate_number_of_terms_in_class(training_data[class_name])
        logging.debug("Class contains " + str(number_of_terms_in_class) + " terms")

        # Taken the product of all the probabilities of a term appearing in the
        # class as calculated during training
        for term in to_classify_terms:

            # Treat captitalized and lowercase as a single term
            term = term.lower()

            # We ignore stop words and term.isalnum():
            if term not in stopwords and term.isalnum():

                # We have to smooth the probabilities of unknown words.  This
                # means that a term we don't recognize is treated as having a
                # very small probability.  If we left it as 1 it doesn't impact
                # the product.  In truth , unrecognized terms should be treated
                # as rare rather than common.  Here we use laplace smoothing(or
                # odd one smoothing)
                if term in training_data[class_name]:
                    term_frequency = float(training_data[class_name][term])
                else:
                    term_frequency = 0.0

                # A probability very near 0
                term_probability_in_trained_class = (term_frequency + 1)/( number_of_terms_in_class + vocabulary_size)

                if args["printProbabilities"]:
                     logging.warn("The word <" + term + "> occurs with frequency " + str(term_frequency) + " and probability " + str(term_probability_in_trained_class))

                # Log probability used in the product to avoid approaching 0 as we multiple small numbers
                log_probability_of_words_in_class += math.log(term_probability_in_trained_class)

            # We now know p(c) and p(w|c).  We are planning the use bayse
            # Theorem: p(A|B) =  p(B|A) * p(A) / p(B) to learn p(c|w) - the log
            # probability of a class given the words in a document.  Plugging
            # info Bayes Theorem: p(c|w) = p(w|c)*p(c)/p(w). p(w) is only a
            # function of words in the document we are classifying, and it
            # therefore can be considered constant across classes.  We can
            # therefore drop it.  So now we have p(c|w) = p(c) * p(w|c)
            class_probabilities[class_name] = log_probability_of_words_in_class + log_probability_of_class

    logging.debug(" ")

    # We now have a bunch of probabilities, one per class.  We simply
    # take the highest probability and label the document as belong to
    # that class.
    max_class = None
    max_probability = -sys.float_info.max
    for class_name, probability in class_probabilities.iteritems():
        logging.debug(" Probability of " + class_name + "is " + str(probability))
        if probability > max_probability:
            max_probability = probability
            max_class = class_name

    return max_class, max_probability




################################################################################
#
# We need to know the number of terms in a class.  We are specifically NOT
# counting UNIQUE terms.  This is essentially rebuilding the number of words in
# a documents from term frequencies.
#
################################################################################
def calculate_number_of_terms_in_class(terms):
    count = 0
    for term,frequency in terms.iteritems():
        count += float(frequency)

    return count




################################################################################
#
# We need to know the entire vocabulary size for the entire corpus domain.  As
# usual, vocabulary size counts the number of unique terms in a corpus.
#
################################################################################

def calculate_vocabulary_size(training_data):
    vocabulary = {}
    for class_name, class_terms in training_data.iteritems():
        for term in class_terms:
            vocabulary[term] = term

        return len(vocabulary)




################################################################################
#
# Pull in the training data from the CSV file.
#
################################################################################

def load_training_data():
    training_data = {}
    class_names = {}

    # Read the CSV file generated during traing - comprised of the corpus(
    # class in this example)name, the term(stemmed or not depending on user inp
    # ut) and the probability of the term occuring within the given class.
    rows = fs.read_csv("bayes_training.csv")

    # Intereate through each of the rows in the CSV file
    for index, [category_name,term,probability] in enumerate(rows):

        # Skip the header
        if index > 0:
            # Store the probability for each term
            if category_name in training_data:
                training_data[category_name][term] = probability
            else:
                training_data[category_name] = {}

            class_names[category_name] = category_name
    return training_data,[name for name in class_names.keys()]




################################################################################
#
# We train the classifier using one or more corpora(throuth one corpora would
# be pointless).  Training results in a CSV file that capture the class, term
# term frequency of each term in all training corpora.  We exclude stop words
# and punctuation from all calculations.
#
################################################################################

def train_classifier(args):
    logging.debug("Training classifier")

    training_corpora = {}

    # Use the same corpora that we have used in previous demos
    training_set_names = ["abc","genesis","gutenberg","inaugural","stateUnion",
                        "webtext","custom"]

    # Open a CSV file with 3 columns.  First column is the name of the corpus
    # (which in this examples is also the name of the class).  The seconde is a
    # single term from the corpus.  Third is the probability with which the term
    # occurs.
    training = fs.open_csv_file("bayes_training.csv",["class","term","probability"]);

    # Ignore stopwords
    stopwords =  nltk.corpus.stopwords.words('english')
    print args
    # Iterate through each of the training sets
    for training_set_name in training_set_names:
        #load the words and corpus name from the requested corpus
        if args[training_set_name]:
            terms_array, corpus_name = words.load_text_corpus({training_set_name:args[training_set_name]})

        # stem the terms if stemming is enabled
        if args["stemming"]:
            terms_array = words.stem_words_array(terms_array)

        # Count up the unique terms in the words array
        terms_counts = words.collect_term_counts(terms_array)

        # Get the total number of words in entire corpus
        num_words = float(len(terms_array))

        # Write the frequency of each term occuring in the given out to the CSV
        for term, count in terms_counts.iteritems():
            # We ignore stop words and punctuation
            if term not in stopwords and term.isalnum():
                training.writerow([corpus_name,term.lower(),count])



################################################################################
#
# Build the commandline parser for the script and return a map of the entered
# options.  I"n addition, setup logging level based on the user's entered log
# level.  Sepecific options are documented inline.
#
################################################################################

def configure_command_line_arguments():
    # Initialize the commandline argument parser
    parser = argparse.ArgumentParser(description = 'Naive Bayes Classifier')

    # Configure the log level parser.  Verbose shows some log, veryVerbose shows
    # more
    logging_group = parser.add_mutually_exclusive_group(required = False)
    logging_group.add_argument("-v",
                               "--verbose",
                               help = 'Set the log level verbose',
                               action = 'store_true',
                               required = False)

    logging_group.add_argument("-vv",
                               "--veryVerbose",
                               help = "Set the log level verbose.",
                               action = 'store_true',
                               required = False)

    # NLTK supports six built in plaintext corpora.  This allows the user to
    # choose between those six corpora or a seventh option -the corpus the user
    # provided.  THis first is a corpus taken from ABC news.
    parser.add_argument('-abc',
                       '--abc',
                       help="ABC news corpus",
                       required=False,
                       action='store_true')

    # The second corpus is the book of Genesis
    parser.add_argument('-gen',
                       '--genesis', help="The book of Genesis from the Bible.",
                       required=False,
                       action='store_true')

    # Third is a collection of text from project Gutenberg
    parser.add_argument('-gut',
                       '--gutenberg', help="Text from Project Gutenberg.",
                       required=False,
                       action='store_true')

    # Fourth is text from presidential inaugural addresses
    parser.add_argument('-in',
                       '--inaugural', help="Text from inaugural addresses.",
                       required=False,
                       action='store_true')

    # Fifth is text from the State of the Union
    parser.add_argument('-su',
                       '--stateUnion', help="Text from State of the Union Addresses.",
                       required=False,
                       action='store_true')

    # The final NLTK provided corpus is text from the web
    parser.add_argument('-web',
                       '--webtext', help="Text taken from the web.",
                       required=False,
                       action='store_true')

    # Tell the parser that there is an optional corpuse that can be pulled in.
    # The dictionary can contain multiple files and directories(if the user
    # also passes --recursive)
    fs.add_filesystem_path_args(parser,
                                '-c',
                                '--custom',
                                help='Directory of files to include in a custom corpus.',
                                required=False)

    parser.add_argument('-t',
                        '--train',
                        help="Train the classifier using the NLTK tokens",
                        required=False,
                        action='store_true')

    parser.add_argument('-cl',
                        '--classify',
                        help="Classify the contents of classify.txt",
                        required=False)

    parser.add_argument('-s',
                        '--stemming', help="Stem in the classifier or trainer.",
                        required=False,
                        action='store_true')

    parser.add_argument('-lp',
                        '--printProbabilities', help="Print each word probability.",
                       required=False,
                        action='store_true')

    # Parse the passed commandline args and turn them into a dictionary.
    args = vars(parser.parse_args())

    # Configure the log level based on passed in args to be one of DEBUG, INFO,
    # WARN, ERROR, CRITICAL
    log.set_log_level_from_args(args)

    return args


###############################################################################
#
# This is a pythonism.  Rather than putting code directly at the "root"
# level of the file we instead provide a main method that is called
# whenever this python script is run directly.
#
###############################################################################

if __name__ == "__main__":
    main()
