# DoA-Zero-EEG

See the wiki for a detailed description of the project and the data.

## Setting up the Python environment

Please use the following steps to set up your Python env.

We use [rye](https://rye-up.com/) to manage everything.

1. Install `rye` following the instructions on the official website.

2. Clone this repository. (you need to setup your SSH key on the git web
   interface)

3. From within the repository, just type `rye sync`, which will create the
   environment locally in a `.venv` folder within this repository.

   Note that this might take a bit of time because it's going to download
   lots of packages (nvidia, pytorch etc.).

4. Activate the environment with `source .venv/bin/activate`.

5. Run `rye run pre-commit install` to setup pre-commit git hooks, which will be
   triggered everytime you commit.

Then you should be good to go.

In case of any issue, ask them on the
[`#hometown` channel on Mattermost](https://chat.letemple.org/waterloo/channels/town-square).

If you don't have an account on Mattermost, you can create one
[using this link](https://chat.letemple.org/signup_user_complete/?id=w638i4jmypneufekh9prsk369o&md=link&sbr=fa).
