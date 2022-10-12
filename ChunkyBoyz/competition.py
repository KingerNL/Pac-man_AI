import capture
import layout
import textDisplay

import argparse
import collections
import datetime
import email.utils
import inspect
import itertools
import logging
import math
import mimetypes
import multiprocessing
import os
import pickle
import random
import smtplib
import sys
import urllib2
import zipfile
from contextlib import contextmanager
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

DEFAULT_SECRETS = os.path.join(os.path.dirname(__file__), 'hva.secrets')

LOG_FILENAME = "log.txt"

logging.basicConfig(filename=LOG_FILENAME,
        filemode="w+",
        format="%(asctime)s %(module)s %(levelname)s %(message)s",
        level=logging.INFO)

SECRETS_KEYS = [
        ('course_name', "Course name"),
        ('host', "SMTP host"),
        ('port', "SMTP port"),
        ('user', "SMTP username"),
        ('password', "SMTP password"),
        ('name', "Display name"),
        ('instructor_mail', "Comma-separated list of instructor email addresses"),
        ('download_url', "URL to download student data from"),
]

TIMESTAMP_FMT = "%Y-%m-%d.%H-%M-%S"

# GMail has a attachment size limit of 24 MB, in bytes.
ATTACHMENT_SIZE_LIMIT = 24e6

@contextmanager
def silence_stdout():
    """
    Temporarily set the output of print() to nothing.
    """
    new_target = open(os.devnull, 'w')
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def create_secrets(secrets_file=DEFAULT_SECRETS):
    """
    Create a secrets file.

    Instead of storing secret information, such as login credentials, publicly
    in this file, we store them in a separate file.  That way, we can easily
    share this file with others, without accidentally sharing private
    information.
    """
    info = "We are going to make a new secrets file. \n"
    info += "If {} already exsists, we will try to recover information from "
    info += "that file when the reply is empty."
    print(info.format(secrets_file))

    secrets = {}
    try:
        with open(secrets_file, 'r') as f:
            secrets = pickle.load(f)
    except:
        pass
    for key, prompt in SECRETS_KEYS:
        if key in secrets:
            prompt += " (now: \"{}\")".format(secrets[key])
        secrets[key] = raw_input(prompt + ": ").strip() or secrets.get(key, '')
    with open(secrets_file, 'w') as f:
        pickle.dump(secrets, f)


def load_secrets(secrets_file=DEFAULT_SECRETS):
    """
    Load data from a secrets file and check if all required fields are
    present.
    """
    with open(secrets_file, 'r') as f:
        secrets = pickle.load(f)
    missing = []
    for key, _ in SECRETS_KEYS:
        if key not in secrets:
            missing.append(key)
    if missing:
        msg = "Missing values for the following items: {}".format(
                ", ".join(missing))
        raise RuntimeError(msg)
    return secrets


def check_positive(value):
    """
    argparse helper function to make sure values are > 0.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("""{} is negative or zero, but should
        be positive.""".format(value))
    return ivalue


def check_positive_or_zero(value):
    """
    argparse helper function to make sure values are >= 0.
    """
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("""{} is negative, but should be
                positive or zero.""".format(value))
    return ivalue


def check_is_file(value):
    """
    argparse helper function to make sure value is a path to an exisiting
    file.
    """
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError("{} is not a file name".format(value))
    return value


def parse_arguments():
    """
    Parse command line arguments, and provides text for --help.

    These will be in a form that the original game can process.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # competition.py-specific arguments.
    parser.add_argument("-A", "--agent-args", dest="agentArgs",
            help="""Options for each agent in a game (e.g.,
            key1=value1,key2=value2,...)""")
    parser.add_argument("-D", "--no-download", dest="no_download",
            action="store_true",
            help="Do not download new student data.")
    parser.add_argument("-M", "--no-mail", dest="no_mail",
            action="store_true",
            help="Do not notify participants of new results.")
    parser.add_argument("-P", "--no-play", dest="no_play",
            action="store_true",
            help="""Do not play the competition. This may be useful if you
            only want to download student data, upload results, or notify
            students.""")
    parser.add_argument("-S", "--edit-secrets", dest="edit_secrets",
            action="store_true",
            help="Edit or create the secrets file. No competition is played.")
    parser.add_argument("-T", "--threads", dest="threads",
            type=check_positive_or_zero, default=multiprocessing.cpu_count(),
            help="""Number of threads used for running matches.""")

    # capture.py-specific arguments.
    parser.add_argument("-l", "--layout", dest="layout",
            default="defaultCapture",
            help="""The layout file from which to load the map layout; use
            RANDOM for a random maze; use RANDOM<seed> to use a specified
            random seeg, e.g., RANDOM23.""")
    parser.add_argument("-q", "--quiet", dest="display_type",
            action="store_const", const="quiet",
            help="Display minimal output and no graphics.")
    parser.add_argument("-Q", "--super-quiet", dest="display_type",
            action="store_const", const="super_quiet",
            help="Same as -q but agent output is also suppressed.")
    parser.add_argument("-i", "--time", dest="length",
            type=check_positive, default=1200,
            help="Time limit of a game in moves.")
    parser.add_argument("-n", "--numGames", dest="numGames",
            type=check_positive, default=1,
            help="Number of games to play.")
    parser.add_argument("-f", "--fixRandomSeed", dest="fixRandomSeed",
            action="store_true",
            help="Fixes the random seed to always play the same game.")
    parser.add_argument("--record", dest="record",
            action="store_true", default=True,
            help="Writes game histories to a file (named RedTeam-BlueTeam).")
    parser.add_argument("-x", "--num-training", dest="numTraining",
            type=check_positive_or_zero, default=0,
            help="How many episodes are training (suppresses output).")
    parser.add_argument("-c", "--catch-exceptions", dest="catchExceptions",
            action="store_true", default=True,
            help="Catch exceptions and enforce time limits.")
    parser.add_argument("-s", "--secrets", dest="secrets",
            type=check_is_file, default=DEFAULT_SECRETS,
            help="File containing the 'secret' infomation.")

    args = parser.parse_args()

    if args.display_type == "quiet":
        args.display_fn = textDisplay.NullGraphics
    elif args.display_type == "super_quiet":
        args.display_fn = textDisplay.NullGraphics
        args.muteAgents = True
    else:
        args.display_fn = textDisplay.PacmanGraphics

    if args.layout == "RANDOM":
        args.layout_type = ("random", None)
    elif args.layout.startswith("RANDOM"):
        seed = args.layout[6:]
        if seed.isdigit():
            seed = int(seed)
        args.layout_type = ("random", seed)
    else:
        l = layout.getLayout(args.layout)
        if l is None:
            raise Exception("The layout '{}' cannot be found in "
                    "folder layouts/".format(args.layout))
        args.layout_type = ("map", l)

    agent_args = capture.parseAgentArgs(args.agentArgs)
    args.agentArgs = agent_args

    args.delay_step = 0
    return args


def update_arguments(args, red_name, red_agents, blue_name, blue_agents):
    """
    Give a matchup-specific update on the command line arguments.

    Again, these will be in a form that the original game can process.
    """
    # This will be a list of agents [r1, b1, r2, b2, ...].
    agents = sum([list(el) for el in zip(red_agents, blue_agents)], [])

    layouts = []
    for _ in range(args.numGames):
        if args.layout_type[0] == "random":
            seed = args.layout_type[1]
            l = layout.Layout(capture.randomLayout(seed).split('\n'))
        else:
            l = args.layout_type[1]
        layouts.append(l)

    vargs = vars(args)
    return {
            "layouts": layouts,
            "agents": agents,
            "display": args.display_fn(),
            "length": args.length,
            "numGames": args.numGames,
            "record": args.record,
            "numTraining": args.numTraining,
            "redTeamName": red_name,
            "blueTeamName": blue_name,
            "muteAgents": vargs.get("muteAgent", False),
            "catchExceptions": vargs.get("catchExceptions", False),
            "delay_step": vargs.get("delay_step", 0),
            }


class Result:
    """
    Represents both the game outcomes and the corresponding score multipliers.
    """
    WIN = 5
    TIE = 3
    LOST = 1
    ERROR = -6

    @classmethod
    def get_name(cls, number):
        """
        Get a string representation of a Result value.
        """
        for member in dir(cls):
            if getattr(cls, member) == number:
                return member
        return "UNKNOWN"


class StudentError:
    """
    Disqualifying errors and their string representation.
    """
    CannotImport = "Error during import, perhaps a SyntaxError?"
    NoCreateTeam = "No createTeam function found."
    CreateTeamTooFewArgs = "createTeam has too few arguments."
    CreateTeamTooManyNondefaults = "createTeam has too many arguments without parameters."
    CreateTeamRuntime = "createTeam ran into a RuntimeException."
    CreateTeamNoReturn = "createTeam returned nothing or None."
    createTeamWrongReturn = "createTeam did not return an indexable collection of 2 objects."


class Record:
    """
    Stores the frequencies of a team's game outcomes.
    """
    def __init__(self, win=0, tie=0, lost=0, error=0, points=0):
        self.win = win
        self.tie = tie
        self.lost = lost
        self.error = error
        self.points = points

    def score(self):
        """
        Computes the score of this team.

        The score is a sum product of the result frequencies with their
        corresponding score multipliers.
        """
        return ((Result.WIN * self.win)
                + (Result.TIE * self.tie)
                + (Result.LOST * self.lost)
                + (Result.ERROR * self.error))

    def update(self, points, result=None):
        """
        Stores an occurance of a Result type.
        """
        if result == Result.WIN or result is None and points > 0:
            self.win += 1
        elif result == Result.TIE or result is None and points == 0:
            self.tie += 1
        elif result == Result.LOST or result is None and points < 0:
            self.lost += 1
        elif result == Result.ERROR:
            self.error += 1
        else:
            raise RuntimeError("Unknown result: {}".format(result))
        self.points += points

    def __repr__(self):
        """
        Gives a human-readable representation of Record.
        """
        name = self.__class__.__name__
        return ("{n}(win={r.win}, tie={r.tie}, lost={r.lost}, "
                "error={r.error}, points={r.points})".format(n=name, r=self))

    def __cmp__(self, other):
        """
        Makes Records sortable.
        """
        if type(self) != type(other):
            return -1
        if self.score() != other.score():
            return cmp(self.score(), other.score())
        return cmp(self.points, other.points)

    def __add__(self, other):
        """
        Implements operator + for 2 Record objects.

        """
        if type(self) != type(other):
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'"
                    .format(self.__class__, other.__class__))
        return self.__class__(win=self.win+other.win,
                tie=self.tie+other.tie,
                lost=self.lost+other.lost,
                error=self.error+other.error,
                points=self.points+other.points)


class Scoreboard:
    def __init__(self):
        self.disqualified_teams = {}
        self.participating_teams = {}
        self.records = collections.defaultdict(
                lambda: collections.defaultdict(Record))
        self._lock = multiprocessing.Lock()

    def disqualify(self, teams):
        """
        Registers all teams disqualified by initial inspection.

        teams is a dictionary.  The keys should be the teams' names, and its
        values are strings with a human-readable description of the error.
        """
        with self._lock:
            self.disqualified_teams.update(teams)

    def register_participants(self, teams):
        """
        Register a collection of team names as participants.

        teams is a dictionary.  The keys should be the teams' names, and its
        values are the factory to create the team's agents.
        """
        with self._lock:
            self.participating_teams.update(teams)

    def add_result(self, leftTeam, rightTeam, points,
            leftResult=None, rightResult=None):
        """
        Adds scores to the score board.
        """
        if leftResult is None and rightResult is None:
            logging.info("Adding result for {} vs {}: {}".format(
                leftTeam, rightTeam, points))
        else:
            logging.info("Adding result for {} ({}) vs {} ({}): {}".format(
                leftTeam, Result.get_name(leftResult),
                rightTeam, Result.get_name(rightResult),
                points))
        with self._lock:
            self.records[leftTeam][rightTeam].update(points, leftResult)
            self.records[rightTeam][leftTeam].update(-points, rightResult)

    def ranking(self):
        """
        Returns a list of team names, sorted by highest-ranking teams first.
        """
        totals = []
        for team in self.records.keys():
            totals.append((team, sum(self.records[team].values(), Record())))
        return sorted(totals, key=lambda x: x[1], reverse=True)

    def make_pairings(self):
        """
        Generate all game pairings of all participants.

        Each team plays against each other team a single time.  The team's home
        side is randomized.

        The returned value is a multiprocessing.Queue.
        """
        def _shuffled(x):
            y = list(x)
            random.shuffle(y)
            return y
        combinations = itertools.combinations(self.participating_teams.items(), 2)
        queue = multiprocessing.Queue()
        for match in combinations:
            queue.put(_shuffled(match))
        return queue

    @property
    def participants(self):
        """
        Returns an alphabetically sorted list of team names.
        """
        return sorted(list(self.participating_teams.keys()))


def generate_html_report(scoreboard, report_file, courseName, timestamp_start, timestamp_finish, **args):
    """
    Make an HTML document that will represent the score in scoreboard.
    """
    fmt = {}
    fmt['courseName'] = courseName
    fmt['timestamp_start'] = timestamp_start
    fmt['timestamp_finish'] = timestamp_finish
    fmt['argv'] = sys.argv

    fmt['title'] = "{} Capture the Flag results of {:%d-%m-%Y}".format(courseName, timestamp_start)
    fmt['duration'] = timestamp_finish - timestamp_start
    fmt['Result'] = Result
    fmt['disqualified_teams'] = ""
    for team, error in scoreboard.disqualified_teams.items():
        fmt['disqualified_teams'] += "<li><strong>{}</strong>: {}</li>".format(
                team, error)

    if len(fmt['disqualified_teams']):
        fmt['disqualified_teams'] = """
<p class="disqualified">The following teams are disqualified by initial inspection:
</p>
  <ul class="disqualified">
    {}
  </ul>""".format(fmt['disqualified_teams'])

    fmt['game_outcomes'] = ''
    if scoreboard.participants:
        fmt['ranking'] = ""
        for i, (n, r) in enumerate(scoreboard.ranking()):
            s = r.score()
            fmt['ranking'] += """    <tr>
          <td>{i}</td>
          <td>{n}</td>
          <td class="points">{r.points}</td>
          <td class="win">{r.win}</td>
          <td class="tie">{r.tie}</td>
          <td class="lost">{r.lost}</td>
          <td class="error">{r.error}</td>
          <td class="score">{s}</td>
        </tr>
    """.format(i=i+1, n=n, r=r, s=s)

        participants = scoreboard.participants
        header = ''
        subheader = len(participants) * """
          <th scope="col" class="points">Points</th>
          <th scope="col" class="win">Win</th>
          <th scope="col" class="tie">Tie</th>
          <th scope="col" class="lost">Lost</th>
          <th scope="col" class="error">Error</th>
    """
        fmt['game_outcomes'] = """
<h2>Team ranking</h2>
The team ranking is based on a total competition score.  Each result type is
worth a number of points, as can be seen in the table below.  In case of equal
competition scores, teams with more points collected will be ranked higher.
<table>
  <thead>
    <tr>
      <th>Outcome type</th>
      <th>Competition points</th>
    <tr>
  </thead>
  <tbody>
    <tr class="win">
      <td>Win</td>
      <td>{Result.WIN}</td>
    </tr>
    <tr class="tie">
      <td>Tie</td>
      <td>{Result.TIE}</td>
    </tr>
    <tr class="lost">
      <td>Lost</td>
      <td>{Result.LOST}</td>
    </tr>
    <tr class="error">
      <td>Error</td>
      <td>{Result.ERROR}</td>
    </tr>
  </tbody>
</table>

<table>
  <caption>Raking of all qualified teams.</caption>
  <thead>
    <tr>
      <th>Position</th>
      <th>Team name</th>
      <th>Points</th>
      <th>Win</th>
      <th>Tie</th>
      <th>Lost</th>
      <th>Failed</th>
      <th>Competition score</th>
    </tr>
  </thead>
  <tbody>
{ranking}
  </tbody>
</table>
""".format(**fmt)


    if len(scoreboard.participants) > 1:
        results = ''

        for team in participants:
            header += '    <th scope="colgroup" colspan="5">{}</th>\n'.format(team)
            results += '    <tr>\n'
            results += '      <th scope="row">{}</th>\n'.format(team)
            for rival in participants:
                if rival == team:
                    results += '      <td colspan="5">&mdash;</td>\n'
                else:
                    r = """
          <td class="points">{r.points}</td>
          <td class="win">{r.win}</td>
          <td class="tie">{r.tie}</td>
          <td class="lost">{r.lost}</td>
          <td class="error">{r.error}</td>\n"""
                    results += r.format(r=scoreboard.records[team][rival])
            results += '    </tr>\n'
        fmt['results'] = """
        <tr>
          <td rowspan="2"></td>
          {header}
        </tr>
        <tr>
    {subheader}
        </tr>
    {results}""".format(header=header, subheader=subheader, results=results)
        fmt['game_outcomes'] += """

<h2>Results per matchup</h2>
<table>
  <caption>Results per competitor.</caption>
{results}
</table>""".format(**fmt)


    with open(report_file, 'w') as f:
        f.write("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
.disqualified {{
  color: red;
}}
td, th {{
  text-align: center;
}}

th[scope="row"] {{
  text-align: right;
}}

.win {{
  background-color: lightgreen;
}}
.tie {{
  background-color: lavender;
}}
.lost {{
  background-color: lightblue;
}}
.error {{
  background-color: lightsalmon;
}}
.points {{
  background-color: lightgoldenrodyellow;
}}

td[colspan] {{
  background-color: lightgrey;
}}

caption {{
  font-style: italic;
}}
</style>
<title>{title}</title>
<meta name="generator" content="Julian and Patricks awesome competition script!">

</head>
<body>
<h1>{title}</h1>

<p>These are the results for a run of the Capture the Flag
competition, part of the course {courseName}. The simulation
started at {timestamp_start:%d-%m-%Y, %H:%M:%S}, and ended at
{timestamp_finish:%d-%m-%Y, %H:%M:%S}.
Running all simulations took {duration} time.</p>

{disqualified_teams}

{game_outcomes}
<!-- This competition has been run with the following arguments:
     {argv}
-->
</body>
</html>""".format(**fmt))


def download_student_data(secrets):
    """
    Download student files from a SurfDrive folder.

    To set up this folder in SurfDrive, you have to do create two different
    "Public Links" (on the SurfDrive website) for a folder named "students".

    One link should allow others to "Upload only (File Drop)" to the folder.
    By sharing this link, students can upload their single-file Python script
    to this folder.

    Another link should allow others to "Download / View" the folder.  This is
    used in this part of the code.  Do not share this link with students, as
    they enables them to look at other teams' code.
    """
    url = secrets['download_url']
    if url.startswith('https://surfdrive') and not url.endswith('/download'):
        url += '/download'
    response = urllib2.urlopen(url)
    with open('students.zip', 'wb') as f:
        f.write(response.read())
        logging.info("Downloading student data succesful.")
    zf = zipfile.ZipFile('students.zip')
    for info in zf.infolist():
        filename = info.filename
        if filename == "__init__.py" or filename.endswith(".pyc"):
            continue
        elif filename.endswith('/'):
            if not os.path.exists(filename):
                os.mkdir(filename)
        else:
            with open(filename, 'wb') as f:
                f.write(zf.read(filename))
    logging.info("Unpacking student data succesful.")


def get_student_modules(module):
    """
    Retrieve all submodules from a module.
    """
    submodules = []
    for item in dir(module):
        if item.startswith('_'):
            continue
        member = getattr(module, item)
        if isinstance(member, type(module)):
            submodules.append(member)
    return submodules


def analyse_student_modules(student_modules):
    """
    Find out which modules work as expected.

    This function will return three values:
    - a set of email addresses to send feedback to
    - a list of agent factories contained in the modules that stick to the
      rules.
    - a list of ModuleErrors for modules not sticking to the rules.
    """
    email_addresses = set()
    agent_factories = {}
    disqualified_teams = {}
    for student_module in student_modules:
        module_name = student_module.__name__
        team_name = '.'.join(module_name.split('.')[1:])
        error = None

        if hasattr(student_module, "CONTACT"):
            student_contact = getattr(student_module, "CONTACT")
            if type(student_contact) == str:
                email_addresses.update(student_contact.split(','))
            elif hasattr(student_contact, "__iter__"):
                for contact in student_contact:
                    if type(contact) == str:
                        email_addresses.update(contact.split(","))

        if not hasattr(student_module, "createTeam"):
            error = StudentError.NoCreateTeam
        else:
            createTeam = getattr(student_module, "createTeam")
            team_red = None
            team_blue = None
            try:
                with silence_stdout():
                    team_red = createTeam(0, 1, True)
                    team_blue = createTeam(2, 3, False)
            except:
                arg_spec = inspect.getargspec(createTeam)
                nondefault_args = len(arg_spec.args) - len(arg_spec.defaults)
                if len(arg_spec.args) < 3:
                    error = StudentError.CreateTeamTooFewArgs
                elif len(arg_spec.args) - len(arg_spec.defaults) > 3:
                    error = StudentError.CreateTeamTooManyNondefaults
                else:
                    error = StudentError.CreateTeamRuntime

            if None in (team_red, team_blue) and error is not None:
                error = StudentError.CreateTeamNoReturn
            else:
                try:
                    if len(team_red) != 2 or len(team_blue) != 2:
                        raise TypeError(StudentError.CreateTeamWrongReturn)
                except:
                    error = StudentError.CreateTeamWrongReturn
        if error:
            disqualified_teams[team_name] = error
        else:
            agent_factories[team_name] = createTeam

    return email_addresses, agent_factories, disqualified_teams


def select_file(directory, extension):
    """
    Select a file with extension in directory.

    If multiple matches are present, ask the user what to do.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files.append(filename)
    if not files:
        msg = "No files with extension {} found in {}".format(extension, directory)
        logging.error(msg)
        raise ValueError(msg)
    filename = files[0]
    if len(files) > 1:
        print("Multiple matching files present:")
        files.sort()
        padding = int(math.log10(len(files))) + 1
        fmt = "{{:{}d}}.  {{}}".format(padding)
        for i, f in enumerate(files):
            print(fmt.format(i, f))
        file_index = int(raw_input("Choose your zip to upload: "))
        if file_index < 0 or file_index >= len(files):
            raise IndexError("Your input i should be 0 <= i < {}".format(len(files)))
        filename = files[file_index]
    return os.path.join(directory, filename)


def zip_results(output_dir, remove_src=False):
    """
    Create a zip archive of the results.html and replay files.
    """
    zip_name = output_dir + ".zip"
    archive_dir = output_dir.split(os.sep)[-1]
    zf = zipfile.ZipFile(zip_name, mode='w')
    for f in os.listdir(output_dir):
        full_name = os.path.join(output_dir, f)
        archive_name = os.path.join(archive_dir, f)
        zf.write(os.path.join(output_dir, f), archive_name)
    zf.close()

    if remove_src:
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)

    return output_dir + ".zip"


class MethodRequest(urllib2.Request):
    """
    Enables urllib2.Request with other HTTP methods than GET or PUT.

    Copied from https://gist.github.com/logic/2715756
    """
    def __init__(self, *args, **kwargs):
        if 'method' in kwargs:
            self._method = kwargs['method']
            del kwargs['method']
        else:
            self._method = None
        return urllib2.Request.__init__(self, *args, **kwargs)

    def get_method(self, *args, **kwargs):
        if self._method is not None:
            return self._method
        return urllib2.Request.get_method(self, *args, **kwargs)


def upload_file(filename, remove_local=False):
    """
    Upload a file to https://transfer.sh.
    """
    logging.info("Filename to upload: '{}'".format(filename))
    basename = filename.split(os.path.sep)[-1]
    url = "https://transfer.sh/{}".format(basename)
    mimetype = mimetypes.guess_type(filename)[0]
    filesize = int(os.stat(filename).st_size)
    headers = {'Content-type': mimetype,
            'Content-length': filesize}
    with open(filename, 'r') as f:
        request = MethodRequest(url, f, headers, method='PUT')
        response = urllib2.urlopen(request)
    upload_url = response.read()
    logging.info("Uploaded file is available at {}".format(upload_url))

    if remove_local:
        os.remove(filename)

    return upload_url


def notify_students_by_mail(email_addresses, timestamp, results_file, secrets):
    """
    Send the results file to email_addresses, in an appropriate way.

    The results file size may be over the attachment size limit of most SMTP
    servers.  Therefore, there is a file size check.  If the results file is
    larger than that limit, it will be uploaded to transfer.sh.

    Partially based on https://stackoverflow.com/a/3363254
    """
    sender = "{name} <{user}>".format(**secrets)
    instructor_mail = secrets['instructor_mail'].split(',')
    receivers = list(email_addresses) + instructor_mail
    to_mail = ', '.join(list(map(lambda s: "<{}>".format(s), instructor_mail)))

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = to_mail
    msg['Date'] = email.utils.formatdate(localtime=True)
    msg['Subject'] = "[{}] Results of {:%d %B, %H:%M}".format(secrets['course_name'],
            timestamp)

    if os.stat(results_file).st_size < ATTACHMENT_SIZE_LIMIT:
        logging.info("Trying to send results file as attachment.")
        with open(attachment, 'rb') as data:
            part = MIMEApplication(data.read(), Name=attachm_name)
        part['Content-Disposition'] = 'attachment; filename="{}"'.format(attachm_name)
        msg.attach(part)
        extra_text = "The results are attached."
    else:
        logging.info("Trying to upload results file to transfer.sh.")
        file_url = upload_file(results_file, remove_local=True)
        extra_text = "The results are available at: {}".format(file_url)

    msg.attach(MIMEText("""Hi,

You have previously uploaded an agent to the course {course_name}.  I just finished computing the results of all agents competing against each other.  {extra_text}

Best,
AI Capture the Flag Competition Bot"""))
    message = msg.as_string().format(
            sender=sender, to=to_mail,
            course=secrets['course_name'], time=timestamp,
            extra_text=extra_text)
    smtp = smtplib.SMTP(secrets['host'], secrets['port'])
    smtp.starttls()
    smtp.login(secrets['user'], secrets['password'])
    smtp.sendmail(sender, receivers, message)
    smtp.quit()


def run_match(args, output_dir, match_queue, results):
    """
    Worker for multithreading, runs a single match at a time.

    This function will prepare everything to make a call to capture.runGames
    for the matches in match_queue, a multiprocessing.Queue.  Results of
    capture.runGames are put into results, another multiprocessing.Queue.
    """
    while True:
        try:
            (red_name, red_factory), (blue_name, blue_factory) = match_queue.get(timeout=1)
            logging.info("Playing {} vs {}".format(red_name, blue_name))
            red_result = Result.WIN
            blue_result = Result.WIN

            if args.fixRandomSeed:
                random.seed('cs188')

            # Create agents, check on errors in this part.
            red_agents = None
            blue_agents = None
            try:
                with silence_stdout():
                    red_agents = red_factory(0, 2, True, **args.agentArgs)
            except:
                red_result = Result.ERROR
            try:
                with silence_stdout():
                    blue_agents = blue_factory(1, 3, False, **args.agentArgs)
            except:
                blue_result = Result.ERROR

            if Result.ERROR in (red_result, blue_result):
                err_msg = "An error occured during {}'s createTeam"
                if red_result == Result.ERROR:
                    logging.info(err_msg.format(red_name))
                if blue_result == Result.ERROR:
                    logging.info(err_msg.format(blue_name))
                logging.info("{} vs {} ended in {}-{} with {} pts".format(
                    red_name, blue_name, red_result, blue_result, 0))
                for _ in range(args.numGames):
                    results.put((red_name, blue_name, 0, red_result, blue_result))
                continue

            # Play the game!
            _args = update_arguments(args, red_name, red_agents, blue_name, blue_agents)
            with silence_stdout():
                games = capture.runGames(**_args)

            # Update the global score card.
            points = [(game.state.data.score, game.agentCrashed or game.agentTimeout)
                    for game in games]

            for point, error in points:
                if error:
                    if point < 0:
                        red_result = Result.ERROR
                        blue_result = Result.WIN
                    elif point > 0:
                        red_result = Result.WIN
                        blue_result = Result.ERROR
                    logging.info("{} vs {} ended in {}-{} with {} pts".format(
                        red_name, blue_name, red_result, blue_result, 0))
                    results.put((red_name, blue_name, 0,
                        red_result, blue_result))
                else:
                    logging.info("{} vs {} ended with {} pts".format(
                        red_name, blue_name, red_result, blue_result, point))
                    results.put((red_name, blue_name, point))

            # Move all replay-% files to results/red_name-blue_name-%d
            match_name = os.path.join(output_dir, "{}-{}".format(red_name, blue_name))
            replay_prefix = 'replay'
            replay_files = filter(lambda s: s.startswith(replay_prefix + '-'),
                    os.listdir('.'))
            for filename in replay_files:
                new_name = match_name + filename[len(replay_prefix):]
                os.rename(filename, new_name)
        except multiprocessing.queues.Empty:
            break

def run_competition(args):
    """
    Run a competition of capture.runGames, generates a report and notifies
    contestants.

    args is the return value of parse_arguments().  See that function to know
    all the options and their meaning.
    """
    logging.info("Starting.")
    args.timestamp_start = datetime.datetime.now()
    scoreboard = Scoreboard()
    secrets = load_secrets(args.secrets)

    if not args.no_download:
        download_student_data(secrets)
    # You must import students after download_student_data(), because you will
    # probably download new submodules.
    import students
    disqualified_on_import = {m: StudentError.CannotImport
            for m in students._IMPORT_ERRORS}
    scoreboard.disqualify(disqualified_on_import)
    student_modules = get_student_modules(students)
    email_addresses, agent_factories, disqualified_teams = analyse_student_modules(student_modules)
    if args.no_play:
        zip_name = select_file("results", ".zip")
    else:
        scoreboard.disqualify(disqualified_teams)
        scoreboard.register_participants(agent_factories)

        logging.info("Disqualified teams: {}".format(scoreboard.disqualified_teams.keys()))
        logging.info("Qualified: {}".format(scoreboard.participants))
        if len(scoreboard.participants) < 2:
            logging.info("Too few participating agents; no competition is run.")

        try:
            # Will raise OSError if this folder already exists, for example, from
            # previous runs.
            os.mkdir("results")
        except OSError:
            pass
        output_dir = args.timestamp_start.strftime(os.path.join("results",
            TIMESTAMP_FMT))
        try:
            os.mkdir(output_dir)
        except OSError:
            pass

        matches = scoreboard.make_pairings()
        match_runners = []
        results = multiprocessing.Queue()
        for _ in range(args.threads):
            p = multiprocessing.Process(target=run_match,
                    args=(args, output_dir, matches, results))
            match_runners.append(p)
            p.start()

        for p in match_runners:
            p.join()

        while not results.empty():
            scoreboard.add_result(*results.get())

        args.timestamp_finish = datetime.datetime.now()
        report_file = os.path.join(output_dir, "report.html")
        generate_html_report(scoreboard, report_file, secrets['course_name'],
                **vars(args))
        zip_name = zip_results(output_dir, remove_src=True)

    if not args.no_mail:
        if not email_addresses:
            logging.info("No email addresses provided, so no email is sent.")
        else:
            notify_students_by_mail(email_addresses, args.timestamp_start,
                    zip_name, secrets)
            logging.info("Email with attachment is sent to the participants.")
    logging.info("Done!")


if __name__ == "__main__":
    args = parse_arguments()
    run_competition(args)
    sys.exit()
    if args.edit_secrets:
        create_secrets(args.secrets)
    else:
        try:
            run_competition(args)
        except Exception as e:
            logging.error("{}".format(e))
