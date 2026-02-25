from pynput import keyboard


class KeyboardController:
    def __init__(self, max_v, max_w):
        """
        :param robot: robot instance
        """

        self.velocity = max_v
        self.angular = max_w

        self.key_states = {
            keyboard.Key.up: False,
            keyboard.Key.down: False,
            keyboard.Key.left: False,
            keyboard.Key.right: False
        }

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        if key in self.key_states:
            self.key_states[key] = True

    def on_release(self, key):
        if key in self.key_states:
            self.key_states[key] = False

    def step(self):
        v, omega = 0.0, 0.0
        if self.key_states[keyboard.Key.up]:
            v = self.velocity
        if self.key_states[keyboard.Key.down]:
            v = -self.velocity
        if self.key_states[keyboard.Key.left]:
            omega = self.angular
        if self.key_states[keyboard.Key.right]:
            omega = -self.angular

        return v, omega


    def stop(self):
        self.listener.stop()

