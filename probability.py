# The MIT Liscense
# Copyright(c) 2015 Thoughtly, Corp
#
# Code: url = https://github.com/pthomas1/MLBlog/blob/master/words.py
# Toturial url = http://www.thoughtly.co/blog/working-with-text/
#
# Description: Lesson 2: Probability
#
#
#
#
# Argparse is a standard python mechanism for handling commandline args while
# avoiding a buncho of boilerplate code.
import argparse

# This is a moudle that provides a bunch of simple methods that make accessing
# the filesystem simpler
from utils import fs, charting, log

# python logging allows us to log formatted log messages at different log levels
import logging

# numpy is just used for some simple array helpers
import numpy

# need to generate pseudo random numbers
import random

def main():

    # Build the commandline parser and return enrere args. This also setups
    # up any non-ML/NLP config needed by the script (such as logging)
    args = configure_command_line_arguments()

    random.seed()

    num_trials = int(args["numTrials"])
    print args
    # execute the coin flip test
    if args['coinFlip']:
        generate_coin_flip_distribution_offset(num_trials, float(args["coinFlipMultiplier"]))

    # execute the dice roll text
    if args["diceRoll"]:
        generate_dice_roll_sum_distribution(num_trials,int(args["numDice"]))

    # generate a gaussian distribution
    if args["gaussianDistribution"]:
        generate_gaussian_distribution_pdf(num_trials,float(args["mean"]),float(args["standardDeviation"]))

    # generate a uniform distribution
    if args["uniformDistribution"]:
        generate_uniform_distribution_pdf(num_trials)

    # generate a poisson distribution
    if args["poissonDistribution"]:
        generate_poisson_distribution(num_trials, int(args["lambda"]))

    if args["jars"]:
        marbles_and_jars(num_trials)


################################################################################
#
# This function is responsible for running a simulation of draws frome a number
# of jars filled with colored marbles.  The user configure the simulation in
# marbles.csv.  Each row is a jar.  Columns represent the marbles in each of the
# jars.  Two are provided to match the tutorial text, but an arbitrary number
# can be simulated.
#
################################################################################

def marbles_and_jars(num_trials):
    # read from the csv file of jars
    rows = fs.read_csv("marbles.csv")
    logging.debug("Read rows:" + str(rows))

    jars = {}
    headers = []
    marble_picks = {}

    # go through the rows and build a dictionary of jar_name => array of marble
    # color
    for index, row in enumerate(rows):
        # first row is just header data
        if index == 0:
            headers = row
        else:
            # go through each of headers(these are columns)
            for column_index, header in enumerate(headers):
                # if the first colunm  that it is the name of the jar
                # -initialize the array to empty(no marbles)
                if column_index == 0:
                    jars[row[0]] = []
                else:
                    # each other column represents a number of marbles, the name
                    # of the marbles is in the header
                    marble_color = header

                    # initialize the counters for picking marbles for the given
                    # color
                    marble_picks[marble_color] = 0

                    # set blank cells to 0, otherwise add the value in the cell
                    if len(row[column_index]) == 0:
                        num_marbles = 0
                    else:
                        num_marbles = int(row[column_index])

                    # expand an array of colors, 1 element for each num_marbles
                    jars[row[0]] += [marble_color] * num_marbles

    logging.info("Jars: " + str(jars))

    for i in range(0, num_trials):
        # pick a random jar from all of the jars w/out taking the marbles into
        # consideration
        jar_names = jars.keys()
        jar_name = jar_names[random.randint(0,len(jar_names)-1)]

        # now draw a single marble from all the marbles given that we selected
        # a jar
        marbles = jars[jar_name]
        marble = marbles[random.randint(0, len(marbles) - 1)]
        marble_picks[marble] += 1
        logging.info("Marble picks: " + str(marble_picks))

    # prepare the data for plotting
    keys = []
    data = []
    for key, value in marble_picks.iteritems():
       column_name = key + " (" + str(value) + ")"
       keys.extend([column_name])
       data.extend([value/float(num_trials)])

    description_list = []
    for jar_name, jar_marbles in jars.iteritems():
        description_list.append(jar_name + "(" + str(len(jar_marbles)) + ")")
    description = ", ".join(description_list)

   # plot the data
    charting.bar_chart("marbles.png",
                       [data],
                       "Marbles in Jars (" + str(num_trials) + ") - " + description,
                       keys,
                       "Probabilities",
                       None,
                       ['#59799e'])




################################################################################
#
# Generate and plot a Uniform Distribution from zero to one.  The distribution
# is generate by the python randon number generator.
#
################################################################################

def generate_uniform_distribution_pdf(number_of_samples):
    values = []

    # generate uniform distribution random samples
    for i in range(0, number_of_samples):
        values.extend([random.random()])

    # plot the distribution
    charting.plot_distribution("uniform_distribution.png",
                               "Uniform Distribution (" + str(number_of_samples) + ")",
                               "Likelihoods",
                               num_buckets=10,
                               data=values,
                               show_bucket_values=True,
                               color='#59799e',
                               normalize=True);




################################################################################
#
# Generate and plot a Poisson Distribution based on the lambda passed in. The
# distribution is generated by the numpy Poission random number generator,
# generating number_of_samples samples.
#
################################################################################

def generate_poisson_distribution(number_of_samples, lam):

    # gemerate number_of_samples random numbers with a Poisson dist
    values = numpy.random.poisson(lam,number_of_samples).tolist()

    # plot the distribution
    charting.plot_distribution("poisson_distribution.png",
                               "Poisson Distribution - " + str(number_of_samples) + ", lambda = " + str(lam),
                               "Likelihoods",
                               bucket_size=1,
                               data=values,
                               show_bucket_values=True,
                               color='#59799e',
                               normalize=True);





################################################################################
#
# Generate and plot a Gaussian Distribution based on the mean and standard
# deviation passed in.  This distribution is generated by numpy gauss random
# number generateor, generating number_of_samples samples.
#
################################################################################

def generate_gaussian_distribution_pdf(number_of_samples, mean, std_dev):
    values = []

    # generate gaussian distribution with mean and standard deviation
    for i in range(0,number_of_samples):
        values.extend([random.gauss(mean,std_dev)])

    # plot the output
    charting.plot_distribution("Gaussian_Distribution.png",
                               "Gaussian Distribution (" + str(number_of_samples) + ", mean = " + str(mean)+ ", std dev = " +str(std_dev) + ")",
                               "Likelihoods",
                               num_buckets = 10 + int(std_dev * 10),
                               data = values,
                               show_bucket_values= True,
                               color = '#59799e',
                               normalize = True)






################################################################################
#
# Execute N die rolls and output an array of valueso
#
################################################################################

def generate_dice_roll_sum_distribution(number_of_rolls, number_of_dice):
    values = []

    die_or_dice = "Die"
    if number_of_dice > 1: die_or_dice = "Dice"

    logging.info("Rolling " + str(number_of_dice) + " " + str(number_of_rolls)
                 + " times")

    # execute number_of_rolls rolls_of_dice dice and accumulate the sum of the
    # values
    for i in range(0,number_of_rolls):
        sum = 0
        for d in range(0,number_of_dice):
            sum += random.randint(1,6)

        values.extend([sum])

    # plot the output
    charting.plot_distribution("dice_rolls.png",
                               "Roll Distribution - " + str(number_of_dice) +" "+ die_or_dice + ", "+ str(number_of_rolls) + " rolls",
                               "Sum of Values",
                               bucket_size = 1,
                               data = values,
                               show_bucket_values = True,
                               color = '#59799e',
                               normalize = True)




################################################################################
#
# Build the commandline parser for the script and return a map of then entered
# options.  In addition, setup logging based on the user's entered log level.
# Specific options are documented inline.
#
################################################################################

def configure_command_line_arguments():
    # Initialize the commandline argument parser.
    parser = argparse.ArgumentParser(description = 'Play with probabilities')

    # Configure the log level parser.  Verbose shows some logs, veryVerbose
    # shows more
    logging_group = parser.add_mutually_exclusive_group(required = False)
    logging_group.add_argument("-v","--verbose",
                               help = "Set the log level verbose",
                               action = 'store_true',
                               required = False)

    logging_group.add_argument("-vv",
                               "--veryVerbose",
                               help = "set the log level veryVerbose",
                               action = 'store_true',
                               required = False)

    # Run the coin flip simulation and plot the output
    parser.add_argument('-cf',
                        '--coinFlip',
                        help = "Generate the coin flip offset chart",
                        required = False,
                        action = 'store_true')

    # configure how big of a multiplier to use when stepping the coin flipper
    parser.add_argument('-cfm',
                        '--coinFlipMultiplier',
                        help = "Each interaton of the coin flip test runs more times than the previous,where current= cfm * previous.",
                        required = False,
                        default = 1.2)

    # Run the dice roll simulation and plot the output
    parser.add_argument('-d',
                        '--diceRoll',
                        help = "Generate the dice roll distribution",
                        required = False,
                        action = 'store_true')

    # configure the number of dice to use in the dice roller
    parser.add_argument('-nd',
                        '--numDice',
                        help = "How many dice to use in the dice rool distribution",
                        required = False,
                        default = 1)

    # configure the number of trails to use for any of the simulations
    parser.add_argument('-nt',
                        '--numTrials',
                        help = "How many trails to use for generating the distribution",
                        required = False,
                        default = 1000000)

    # Generate a uniform distribution
    parser.add_argument('-ud',
                        '--uniformDistribution',
                        help = "Generate a Uniform distribution",
                        required = False,
                        action = 'store_true')

    # Generate a gaussian distribution
    parser.add_argument('-gd',
                        '--gaussianDistribution',
                        help = "Generate a Gaussian distribution",
                        required = False,
                        action = 'store_true')

    # Set the mean for a gaussian distribution
    parser.add_argument('-m',
                        '--mean',
                        help = "Set the mean of the Gaussian Distribution",
                        required = False,
                        default = 0)

    # Set the standard deviation for the gaussian distribution
    parser.add_argument('-sd',
                        '--standardDeviation',
                        help = "Set the standard Deviation for the Gaussion distribution",
                        required = False,
                        default = 1)

    # Generate a poisson distribution
    parser.add_argument('-pd',
                        '--poissonDistribution',
                        help = "Generate a Poisson Distribution",
                        required = False,
                        action = 'store_true')

    # set lambda for the poisson distribution
    parser.add_argument('-l',
                        '--lambda',
                        help = "set lambda(the expected arrival rate) for the poisson distribution",
                        required = False,
                        default = 3)

    # run marble/jar simulation
    parser.add_argument('-j',
                        '--jars',
                        help = "Calculate probability of marble choices",
                        required = False,
                        action = 'store_true')

    # Parser the passed commandline args and turn them into a dictionary
    args = vars(parser.parse_args())

    # Configure the log level based on passed in args to be one of DEBUG, WARN,
    # ERROR, CRITICAL
    log.set_log_level_from_args(args)
    return args




################################################################################
#
# This method generates a number of series of coin flips.  Each series generates
# number_of_flips flips and plots how far from fair the series of flips ended up
# You pass in the max number of flips in a given trial.  The
# flip_count_multiplier is the step size multiplier from one trial to the next.
# Larger flip_count_multiplier results in fever frials
#
################################################################################

def  generate_coin_flip_distribution_offset(max_number_of_flips,
                                            flip_count_multiplier = 1.1):
    flip_counts = []
    head_percentages = []
    number_of_flips = 2
    while number_of_flips < max_number_of_flips:
        logging.info("Generating " + str(number_of_flips) + " coin flips")

        # Flip the coin over and over and report back the number of heads
        # so we can then determine the ratio of heads
        number_of_heads = flip_a_coin(number_of_flips)
        ration_of_heads = float(number_of_heads) / number_of_flips
        flip_counts.extend([number_of_flips])

        # Whatever number we get, unless it was exactly .5, it was off from the
        # ideal.  Record that offset from the expected so we can plot it.
        error_from_expected = abs(.5 - ration_of_heads)
        head_percentages.extend([error_from_expected])

        # It would take forever to walk from 1 to a million, but it's not too
        # bd if we multiply the number of coin flip trials each time instead of
        # adding.
        number_of_flips = int(number_of_flips * flip_count_multiplier) + 1

    # output a text variation of the generated percentages
    logging.debug(str(flip_counts))
    logging.debug(str(head_percentages))

    # we don't have room to display all number labels, so eliminate all but 8
    x_label_step_size = len(flip_counts) / 8
    for i in range(0,len(flip_counts)):
        if i % x_label_step_size:
            flip_counts[i] = ""

    # now generate a plot
    charting.bar_chart("coin_flip.png", [head_percentages],
                       "Heads Flips - Offset from Ideal (" + str(max_number_of_flips) + ")",
                       flip_counts,
                       "Offset from .5 - Larger is Worse",
                       None,
                       ['#59799e'],
                       0,
                       0,
                       False,
                       .5,
                       "none")




################################################################################
#
# Execute N coin flips and output the number of heads and tails
#
################################################################################

def flip_a_coin(number_of_flips):
    number_of_heads = 0
    for i in range(0, number_of_flips):
        # Return a 0 or a 1.  We'll call 1's heads
        number_of_heads += random.randint(0,1)

    return number_of_heads


###############################################################################
#
# This is a pythonism.  Rather than putting code directly at the "root"
# level of the file we instead provide a main method that is called
# whenever this python script is run directly.
#
###############################################################################

if __name__ == "__main__":
    main()
