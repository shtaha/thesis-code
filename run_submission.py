import argparse
import os

import grid2op
from grid2op.Runner import Runner

from agents.agent_dn import make_agent
from lib.submission_utils import write_file, zip_directory, metadata, read_file


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_dir",
        default="submission-test/submission",
        type=str,
        help="Submission directory.",
    )
    parser.add_argument(
        "--agents_dir", default="agents", type=str, help="Agents directory."
    )
    parser.add_argument(
        "--agent_file", default="agent_dn.py", type=str, help="Agent class Python file."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    contents = read_file(os.path.join(args.agents_dir, args.agent_file))

    write_file(os.path.join(args.submission_dir, "submission.py"), contents)
    write_file(
        os.path.join(args.submission_dir, "__init__.py"),
        "from .submission import make_agent\n",
    )
    write_file(os.path.join(args.submission_dir, "metadata"), metadata())
    zip_directory(args.submission_dir)

    env = grid2op.make("l2rpn_case14_sandbox")
    agent = make_agent(env)
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

    runner.run(nb_episode=1, path_save="logs", pbar=True)

    # command = ["python",
    #            os.path.join(problem_dir, "ingestion.py"),
    #            "--dataset_path", input_data_check_dir,
    #            "--output_path", os.path.join("utils", "res"),
    #            "--program_path", problem_dir,
    #            "--submission_path", args.submission_dir]
    #
    # res_ing = subprocess.run(command, stdout=subprocess.PIPE)
