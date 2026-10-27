from lerobot.felix_jr.setup.head import HEAD, run


class FakeRobot:
    """A follower that records the calls the setup script makes and replays scripted nudge directions."""

    class Bus:
        def ping(self, name):
            return 777

    def __init__(self, nudge_directions):
        self.nudge_directions = list(nudge_directions)
        self.bus = self.Bus()
        self.setup_calls: list[tuple[str, tuple[str, ...]]] = []
        self.nudged: list[str] = []
        self.saved: dict[str, int] | None = None

    def setup_head_motor(self, name, configured=()):
        self.setup_calls.append((name, tuple(configured)))

    def nudge_head(self, name):
        self.nudged.append(name)
        return self.nudge_directions.pop(0)

    def save_head_signs(self, signs):
        self.saved = dict(signs)
        return "head.json"


def run_script(robot, answers):
    prompts: list[str] = []
    replies = iter(answers)

    def ask(prompt):
        prompts.append(prompt)
        return next(replies)

    return run(robot, ask=ask, say=lambda *a: None), prompts


def test_head_order_is_eyes_first_then_neck():
    assert list(HEAD) == ["head_eyes", "head_neck"]


def test_eyes_are_set_up_alone_then_the_neck_is_found_by_scanning_past_the_eyes():
    robot = FakeRobot([1, 1])
    signs, prompts = run_script(robot, ["", "", "y", "", "", "y"])
    assert robot.setup_calls == [("head_eyes", ()), ("head_neck", ("head_eyes",))]
    assert signs == {"head_eyes": 1, "head_neck": 1}
    assert "NECK from the EYES" in prompts[0] and "EYES servo ONLY" in prompts[0]
    assert "Plug the NECK servo back in" in prompts[3]


def test_a_no_answer_flips_the_sign():
    signs, _ = run_script(FakeRobot([1, 1]), ["", "", "n", "", "", "n"])
    assert signs == {"head_eyes": -1, "head_neck": -1}


def test_the_sign_accounts_for_the_raw_direction_that_was_actually_used():
    # the eyes could only move in the raw negative direction, and the user said it went DOWN: so raw
    # negative is down, i.e. the sign of raw positive is -1. For the neck, raw positive was used and said no.
    signs, _ = run_script(FakeRobot([-1, 1]), ["", "", "y", "", "", "n"])
    assert signs == {"head_eyes": -1, "head_neck": -1}


def test_repeat_moves_the_servo_again_and_asks_again():
    robot = FakeRobot([1, 1, 1])
    signs, prompts = run_script(robot, ["", "", "r", "", "y", "", "", "y"])
    assert robot.nudged == ["head_eyes", "head_eyes", "head_neck"]
    assert signs == {"head_eyes": 1, "head_neck": 1}
    assert sum("Did the camera move DOWN" in p for p in prompts) == 2


def test_signs_are_saved_with_the_robot():
    robot = FakeRobot([1, -1])
    signs, _ = run_script(robot, ["", "", "y", "", "", "y"])
    assert robot.saved == signs == {"head_eyes": 1, "head_neck": -1}
