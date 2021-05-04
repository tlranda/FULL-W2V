from itertools import product, count
from os import path
import subprocess, numpy as np, argparse as AP

# SELF-AWARENESS: Script bases some file locations relative to itself
script_home = path.dirname(path.abspath(__file__))+"/"
# Arguments
parser = AP.ArgumentParser(description='Performs parameter sweep operation', formatter_class=AP.RawTextHelpFormatter)
parser.add_argument('--identifier', type=str, default='', help='Unique prefix for saved embeddings (default: None)')
parser.add_argument('--repeat', type=int, default=5, help='Number of times to re-run tests for averaging purposes. (default: 5)')
parser.add_argument('--executable', type=str, default='revamp', help='Name of the executable to run, relative to script home ({0}) or filesystem root (/). (default: revamp)'.format(script_home))
parser.add_argument('--debug', type=int, default=1, help='Level of feedback [0,2] s.t. lower is less verbose. (default: 1)')
parser.add_argument('--model-dir', type=str, default='{0}'.format(script_home), help='Directory where experimental models are saved. (default {0})'.format(script_home))
parser.add_argument('--config', nargs='+', type=str, help="""\
File(s) defining arguments to sweep over.
The first noncomment line of the file should be the nonnegative order of sweeping
(you may repeat the same sweeping ordinal to separate sweeping sets in the same argument space).
All other lines should follow one of the following formats:
<argument>: <value>
*<argument>: <value>

Where <argument> is an exact argument name (including '-' characters).
<Value> should be a single element or comma-separated list of elements representing proper values for the argument to assume.
Prepending '*' to the argument name indicates that all elements in <value> should be checked for file or directory existence,
if one or more elements are not found the script will abort.
All file arguments to be checked should be relative to '/' or '{0}'.
Comments may be included with lines that start with the '#' character.
""".format(script_home))
parser.add_argument('--dry-run', action='store_true', help='Display permutations and sum count instead of running the program.')
parser.add_argument('--echo', action='store_true', help='Display outputs received etc')

tab = '\t'
newline = '\n'

# Make list of strings based on product list
def make_list_from_list_product(liprod):
    build = [""]
    for arg in liprod:
        # Extend for new possibilities
        gap = len(build)
        build *= len(arg)
        for val, index in zip(arg, range(len(arg))):
            for _ in range(int(len(build) / len(arg))):
                build[(index * gap) + _] += val+" "
    # String nice-ening
    for _ in range(len(build)):
        build[_] = build[_].rstrip()
    return build

# Make product of items in list (no equivalent itertools.product behavior known)
def lprod(l, repeat=1):
    pools = [pool for pool in l] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

# Make product of all args (no equivalent itertools.product behavior known)
def lprod_star(*_args, repeat=1, depth=None):
    if depth is None:
        pools = [pool for pool in _args] * repeat
    else:
        pools = _args[depth] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result


# Make unified arg string from product of lists
def unify_list_products(l, repeat=1):
    return [" ".join(_) for _ in lprod(l, repeat=repeat)]

def construct_arg_bay(file_queue):
    arg_bay = {}
    for f in file_queue:
        try:
            abspath = path.abspath(path.normpath(path.join(script_home, './', f)))
            print(f"Opening {abspath} as an arg bay")
            with open(abspath, 'r') as c:
                # Retrieve all non-comment lines (comment lines start with '#' character)
                specs = [_.rstrip() for _ in c.readlines() if not _.startswith('#') and not _.isspace()]
                try:
                    # Determine order of sweep config
                    order = int(specs[0])
                except ValueError as e:
                    print(f"Could not determine order of sweep configuration file '{abspath}' (given {f}). First noncomment line should be an nonnegative number specifying order.")
                    raise e
                # Add into dictionary for ordering
                if order not in arg_bay.keys():
                    arg_bay[order] = []
                # Args for the config
                swept = []
                for arg, linenum in zip(specs[1:], count(1)):
                    # Split and clean
                    try:
                        argname, arglist = arg.split(':', maxsplit=1)
                    except ValueError:
                        print(f"Could not read {abspath} (given {f}), it is likely missing a ':' character on line {linenum+1}")
                        exit(1)
                    arglist = arglist.split(',')
                    for val in range(len(arglist)):
                        arglist[val] = arglist[val].lstrip()
                    # Signals that all values are files/directories and must be checked for existence
                    if argname.startswith('*'):
                        argname = argname[1:]
                        for idx, val in enumerate(arglist):
                            # Relative file path fixing
                            if not val.startswith('/'):
                                val = path.abspath(path.join(script_home, val))
                            if not path.exists(val):
                                raise IOError("Could not find required file or directory '{0}' for argument {1} in sweep configuration file '{2}'".format(val, argname, f))
                            else:
                              print(f"\tIncludes file: {val}")
                            arglist[idx] = val # OVERWRITE AS FIXED IF CHANGED
                    # Make each value have its argument name attached so things can be lists instead of dictionaries
                    for val in range(len(arglist)):
                        arglist[val] = argname+" "+arglist[val]
                    swept.append(arglist)
                arg_bay[order].append(swept)
        except IOError as e:
            print(f"Failed to locate sweep configuration file '{abspath}' given {f}")
            raise e
    # Reverse things to execute in the order the user likely expects them to go
    for key in arg_bay:
        arg_bay[key].reverse()
    return arg_bay

def get_sweep(file_queue):
    arg_bay = construct_arg_bay(file_queue)
    # Determine arg bay key depth combinations
    depth_combos = lprod([[_ for _ in range(len(arg_bay[key]))] for key in arg_bay.keys()])
    # Append argsets for each depth combination
    _args = []
    for combination in depth_combos:
        # Construct the full arg set from the current arg bay depths
        arg_set = []
        for key, index in zip(arg_bay.keys(), range(len(arg_bay.keys()))):
            arg_set.append(arg_bay[key][combination[index]])
        # Construct product of current depth sweep
        curdepth_prod = []
        for x in lprod_star(arg_set):
            # New set of parameters in the depth set
                # lprod_star() wraps in list, but only one item in the first list
            curdepth_prod.append(make_list_from_list_product(x[0]))
        # Make new depth set a manageable list
        new_depth = unify_list_products(curdepth_prod)
        _args.extend(new_depth)
    return _args

# Return best, mean, and variance dictionaries for the given list of dictionaried stats
def get_stats(stat_li, num, config_args):
    best, mean, variance, combined = {}, {}, {}, {}
    mean_count = 0
    # Setup
    keys = set()
    for r in stat_li:
        for k in r.keys():
            if k not in keys:
                keys.add(k)
                best[k] = -2.0
                mean[k] = 0.0
                if k in ['similarity', 'accuracy']:
                    best[k] = [-2.0] * len(stat_li[0][k])
                    mean[k] = []
                variance[k] = []
    # Population
    for run in stat_li:
        for entry in run.keys():
            variance[entry].append(run[entry])
            try:
                if run[entry] > best[entry]:
                    best[entry] = run[entry]
            except TypeError:
                if run[entry][0] > best[entry][0]:
                    best[entry] = run[entry]
            if type(mean[entry]) is not list:
                mean[entry] += run[entry]
            else:
                if len(mean[entry]) == 0:
                    mean[entry] = run[entry]
                else:
                    for _ in range(len(mean[entry])):
                        mean[entry][_] += run[entry][_]
        mean_count += 1
    # Post-process
    for entry in keys:
        try:
            mean[entry] /= mean_count
        except TypeError:
            for _ in range(len(mean[entry])):
                mean[entry][_] /= mean_count
        variance[entry] = np.std(variance[entry], axis=0, dtype=np.float64)
        combined[entry] = []
        try:
            combined[entry].append(best[entry])
        except KeyError:
            combined[entry].append(None)
        try:
            combined[entry].append(mean[entry])
        except KeyError:
            combined[entry].append(None)
        try:
            combined[entry].append(variance[entry])
        except KeyError:
            combined[entry].append(None)
    # Combined = full story
    combined['config_num'] = num
    combined['configuration'] = config_args
    # Return
    return best, mean, variance, combined

# Strips speed out of list of bytes W2V output
def speed_strip(li):
    original_phrase = ['Overall', 'Words', 'Per', 'Second:']
    phrase_index = 0
    for word in li:
        word = word.decode()
        # Trigger phrase consumption to reduce false positives
        if original_phrase[phrase_index:] != []:
            if word == original_phrase[phrase_index]:
                phrase_index += 1
            else:
                phrase_index = 0
        else:
            try:
                val = float(word)
                return val
            except ValueError:
                pass
    raise ValueError("Could not locate speed phrase in output")

# Strips eval numbers out of list of bytes Hyperwords output
def eval_strip(li):
    original_phrase = ['embedding']
    phrase_index = 0
    grace = False
    sim = []
    acc = []
    for word in li:
        word = word.decode()
        # Trigger phrase consumption to reduce false positives
        if original_phrase[phrase_index:] != []:
            if word == original_phrase[phrase_index]:
                phrase_index += 1
            else:
                phrase_index = 0
        else:
            try:
                val = float(word)
                sim.append(val)
                phrase_index = 0
                grace = False
            except ValueError:
                try:
                    val = float(word[:-1])
                    acc.append(val)
                except ValueError:
                    # Grace == False (looking for the embedding name)
                    # Grace == True (thought we skipped the embedding name but evidently not)
                    if grace:
                        grace = False
                        phrase_index = 0
                    else:
                        grace = True
    return sim, acc

def main(args):
    sweep_list = get_sweep(args.config)
    max_configurations = len(sweep_list)

    '''
    Heuristic tries to identify the 'best of everything'
      x = speed
      y = similarity
      z = average of cos_add and cos_mul
      relationship: speedup over 10m wps * similarity * average_accuracy / 30%
        normalizes values a bit and weighs them as expected improvements
    '''
    heuristic = lambda x,y,z: (x/10000000.0)*y*(z/30.0)

    print(f"{max_configurations} unique configurations to be tested")
    config_num = 1

    best_heuristic = {'config_num': 0, 'configuration': [None], 'speed': [0,0,0], 'average_similarity': [0,-2,0], 'average_accuracy': [0,0,0], 'similarity': [], 'accuracy': [], 'score': [0,0,0]}
    fastest = {'config_num': 0, 'configuration': [None], 'speed': [0,0,0], 'average_similarity': [0,-2,0], 'average_accuracy': [0,0,0], 'similarity': [], 'accuracy': [], 'score': [0,0,0]}
    best_similarity = {'config_num': 0, 'configuration': [None], 'speed': [0,0,0], 'average_similarity': [0,-2,0], 'average_accuracy': [0,0,0], 'similarity': [], 'accuracy': [], 'score': [0,0,0]}
    best_accuracy = {'config_num': 0, 'configuration': [None], 'speed': [0,0,0], 'average_similarity': [0,-2,0], 'average_accuracy': [0,0,0], 'similarity': [], 'accuracy': [], 'score': [0,0,0]}
    average = {'config_num': 0, 'configuration': [None], 'speed': 0.0, 'average_similarity': 0.0, 'average_accuracy': 0.0, 'similarity': [], 'accuracy': [], 'score': 0.0}
    average_keys = ['speed', 'average_similarity', 'average_accuracy', 'similarity', 'accuracy', 'score']
    variance = {'speed': [], 'average_similarity': [], 'average_accuracy': [], 'similarity': [], 'accuracy': [], 'score': []}
    for key in average_keys:
        average[key+'_count'] = 0

    for _args in sweep_list:
        config_stats = []
        print(f"Load args: {_args}")
        for repeat in range(1, args.repeat+1):
            config_results = {}
            # ./<exe> -output <model_dir>/config_{num}.words [_args from sweep_list]
            save_embeddings = path.normpath(args.model_dir+f"{args.identifier}config_{config_num}_v{repeat}.words")
            run_args = args.executable+" -output "+save_embeddings+" " + _args
            run_args = run_args.split()
            # Show running status
            if args.debug > 0:
                if args.debug > 1:
                    print(f"Config {config_num} -- {repeat}/{args.repeat} Running: {' '.join(run_args)}")
                else:
                    print(f"Config {config_num} -- {repeat}/{args.repeat} Running", end='\r')
            # Create process and make sure it ran successfully to get output
            train_proc = subprocess.run(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if train_proc.returncode != 0:
                print(f"Config {config_num} -- {repeat}/{args.repeat} -- TRAINING FAILED")
                if train_proc.stdout is not None:
                    captured = train_proc.stdout.decode('utf-8')
                    if captured != '':
                        print(captured)
                if train_proc.stderr is not None:
                    captured = train_proc.stderr.decode('utf-8')
                    if captured != '':
                        print(captured)
                print("FAILURE! Skipping")
                # Break out of retrying this arg set
                config_num += 1
                max_configurations -= 1
                break
            # Attempt to parse results
            try:
                if args.echo:
                  print(train_proc.stdout)
                config_results['speed'] = speed_strip(train_proc.stdout.split())
            except AttributeError:
                try:
                    # Sometimes it goes to stderr instead of stdout because people are dumb
                    if args.echo:
                      print(train_proc.stderr)
                    config_results['speed'] = speed_strip(train_proc.stderr.split())
                except AttributeError as e:
                    print(f"Config {config_num} -- {repeat}/{args.repeat} -- TRAINING FAILED")
                    print("Failed to capture input?")
                    raise e
                except ValueError as e:
                    print(train_proc.stdout.split())
                    raise e
            except ValueError as e:
                print(train_proc.stdout.split())
                raise e
            # Post-processing: Evaluate the results
            post_args = ['python3', path.join(script_home, "../analysis/evalHyperWords.py"), save_embeddings]
            # Show running status
            if args.debug > 0:
                if args.debug > 1:
                    print(f"Config {config_num} -- {repeat}/{args.repeat} Evaluating: {' '.join(post_args)}")
                else:
                    print(f"Config {config_num} -- {repeat}/{args.repeat} Evaluating", end='\r')
            # Create process and make sure it ran successfully to get output
            eval_proc = subprocess.run(post_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if eval_proc.returncode != 0:
                print(f"Config {config_num} -- {repeat}/{args.repeat} -- EVALUATION FAILED")
                print(f"{tab}Speed: {config_results['speed']}")
                if eval_proc.stdout is not None:
                    captured = eval_proc.stdout.decode('utf-8')
                    if captured != '':
                        print(captured)
                if eval_proc.stderr is not None:
                    captured = eval_proc.stderr.decode('utf-8')
                    if captured != '':
                        print(captured)
                print("FAILURE! Skipping")
                # Break out of retrying this arg set
                config_num += 1
                max_configurations -= 1
                break
            # Attempt to parse results
            try:
                if args.echo:
                  print(eval_proc.stdout)
                eval_out = eval_proc.stdout.split()
            except AttributeError:
                try:
                    # Sometimes it goes to stderr instead of stdout because people are dumb
                    if args.echo:
                      print(eval_proc.stderr)
                    eval_out = eval_proc.stderr.split()
                except AttributeError as e:
                    print(f"Config {config_num} -- {repeat}/{args.repeat} -- EVALUATION FAILED")
                    print(f"{tab}Speed: {config_results['speed']}")
                    print("Failed to capture input?")
                    raise e
            try:
                config_results['similarity'], config_results['accuracy'] = eval_strip(eval_out)
            except ValueError as e:
                print(f"Config {config_num} -- {repeat}/{args.repeat} -- EVALUATION FAILED")
                print(f"{tab}Speed: {config_results['speed']}")
                print(f"Failed to convert output:\n{eval_out}")
                raise e
            config_results['average_similarity'] = np.average(config_results['similarity'])
            config_results['average_accuracy'] = np.average(config_results['accuracy'])
            config_results['score'] = heuristic(config_results['speed'], config_results['average_similarity'], config_results['average_accuracy'])
            # Append to stats for this config
            config_stats.append(config_results)
            sim_string = " ".join(["{0:.4f}".format(_) for _ in config_results['similarity']])
            acc_string = " ".join(["{0:5.2f}%".format(_) for _ in config_results['accuracy']])
            print(f"Config {config_num} -- {repeat}/{args.repeat} Completed.")
            print(f"{tab}Speed = {config_results['speed']}{tab}{tab}Similarity = {sim_string}{tab}Accuracy = {acc_string}{tab}Heuristic = {config_results['score']:5.2f}")
        # After all repeats, determine full stats for config
        run_best, run_mean, run_variance, run_combined = get_stats(config_stats, config_num, " ".join(run_args))
        print(f"Config {config_num} -- {' '.join(run_args)}")
        print(f"Means:{newline}{tab}Speed = {run_mean['speed']}{tab}{tab}Similarity = {run_mean['similarity']}{tab}Accuracy = {run_mean['accuracy']}{tab}Heuristic = {run_mean['score']:5.2f}")
        print(f"Std:{newline}{tab}Speed = {run_variance['speed']}{tab}{tab}Similarity = {run_variance['similarity']}{tab}Accuracy = {run_variance['accuracy']}{tab}Heuristic = {run_variance['score']:5.2f}")
        # Update winning candidates
        if run_best['score'] > best_heuristic['score'][1]:
            best_heuristic = run_combined
        if run_best['speed'] > fastest['speed'][1]:
            fastest = run_combined
        if run_best['average_similarity'] > best_similarity['average_similarity'][1]:
            best_similarity = run_combined
        if run_best['average_accuracy'] > best_accuracy['average_accuracy'][1]:
            best_accuracy = run_combined
        # Update averages
        for key in run_mean.keys():
            if key in ['similarity', 'accuracy']:
                if average[key] == []:
                    average[key] = run_mean[key]
                else:
                    for _ in range(len(run_mean[key])):
                        average[key][_] += run_mean[key][_]
            else:
                average[key] += run_mean[key]
            average[key+'_count'] += 1
        # Update variances
        for key in run_variance.keys():
            if key in ['similarity', 'accuracy']:
                continue
            variance[key].append(run_variance[key])
        config_num += 1
    # Final output
    for key in average_keys:
        if type(average[key]) is list:
            for _ in range(len(average[key])):
                average[key][_] /= average[key+'_count']
        else:
            average[key] /= average[key+'_count']
        #variance[key] = np.var(variance[key], dtype=np.float64)
    config_num -= 1
    print(f"Successfully evaluated {max_configurations}/{config_num} configurations ({100.0*max_configurations/config_num:.2f}%)")
    #print(f"Average has heuristic score {0}".format(average['score']))
    #print(f"{tab}Speed {0}\n{tab}Similarity {1}\n{tab}Accuracy {2}%".format(average['speed'], average['similarity'], average['accuracy']))
    '''
    # Speed
    print("The following results are formatted as [BEST, Mean, Standard Deviation]")
    print("Fastest is {0} with setup {1} and speed {2}".format(fastest['config_num'], fastest['configuration'], fastest['speed']))
    sim_string = "[["+" ".join(["{0:.4f}".format(_) for _ in fastest['similarity'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in fastest['similarity'][1]])+ \
                 "], ["+str(fastest['similarity'][2])+"]]"
    acc_string = "[["+" ".join(["{0:.4f}".format(_) for _ in fastest['accuracy'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in fastest['accuracy'][1]])+ \
                 "], ["+str(fastest['accuracy'][2])+"]]"
    print("\tHeuristic score {0}\n\tSimilarity {1}\n\tAccuracy {2}".format(fastest['score'], sim_string, acc_string))
    # Similarity
    sim_string = "[["+" ".join(["{0:.4f}".format(_) for _ in best_similarity['similarity'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in best_similarity['similarity'][1]])+ \
                 "], ["+str(best_similarity['accuracy'][2])+"]]"
    acc_string = "[["+" ".join(["{0:.4f}".format(_) for _ in best_similarity['accuracy'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in best_similarity['accuracy'][1]])+ \
                 "], ["+str(best_similarity['accuracy'][2])+"]]"
    print("Best similarity is {0} with setup {1} and similarity {2}".format(best_similarity['config_num'], best_similarity['configuration'], sim_string))
    print("\tHeuristic score {0}\n\tSpeed {1}\n\tAccuracy {2}".format(best_similarity['score'], best_similarity['speed'], acc_string))
    # Accuracy
    sim_string = "[["+" ".join(["{0:.4f}".format(_) for _ in best_accuracy['similarity'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in best_accuracy['similarity'][1]])+ \
                 "], ["+str(best_accuracy['accuracy'][2])+"]]"
    acc_string = "[["+" ".join(["{0:.4f}".format(_) for _ in best_accuracy['accuracy'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in best_accuracy['accuracy'][1]])+ \
                 "], ["+str(best_accuracy['accuracy'][2])+"]]"
    print("Best accuracy is {0} with setup {1} and accuracy {2}".format(best_accuracy['config_num'], best_accuracy['configuration'], acc_string))
    print("\tHeuristic score {0}\n\tSpeed {1}\n\tSimilarity {2}".format(best_accuracy['score'], best_accuracy['speed'], sim_string))
    # Heuristic
    sim_string = "[["+" ".join(["{0:.4f}".format(_) for _ in best_heuristic['similarity'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in best_heuristic['similarity'][1]])+ \
                 "], ["+str(best_heuristic['accuracy'][2])+"]]"
    acc_string = "[["+" ".join(["{0:.4f}".format(_) for _ in best_heuristic['accuracy'][0]])+ \
                 "], ["+" ".join(["{0:.4f}".format(_) for _ in best_heuristic['accuracy'][1]])+ \
                 "], ["+str(best_heuristic['accuracy'][2])+"]]"
    print("Best heuristic is {0} with setup {1} and score {2}".format(best_heuristic['config_num'], best_heuristic['configuration'], best_heuristic['score'][0]))
    print("\tSpeed {0}\n\tSimilarity {1}\n\tAccuracy {2}".format(best_heuristic['speed'], sim_string, acc_string))
    '''

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.executable.startswith('/'):
        args.executable = path.join(script_home, './', args.executable)
    if not path.exists(args.executable):
        raise IOError("Executable script {0} not found!".format(args.executable))
    if not args.model_dir.startswith('/'):
        args.model_dir = path.normpath(path.join(script_home, './', args.model_dir))+'/'
    if not path.exists(args.model_dir):
        raise IOError("Model directory {0} does not exist!".format(args.model_dir))

    if args.dry_run:
        sweep_list = get_sweep(args.config)
        max_configurations = len(sweep_list)
        for _, it in zip(sweep_list, count(1)):
            print("Config {0}: {1}".format(it, _))
        print("Total: ", max_configurations)
        print("Maximum {0} repetitions for averaging == {1} cycles".format(args.repeat, max_configurations * args.repeat))
    else:
        main(args)

