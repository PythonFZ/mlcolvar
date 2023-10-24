from nicegui import ui

class Var():
    def __init__(self):
        self.value=None
        
    def set_value(self, value):
         self.value = value

    def return_value(self):
        return self.value


variable = Var()

input = ui.input(label='Encoder layers', on_change=lambda: variable.set_value(input.value))

ui.button('Check CV model', on_click=lambda: ui.label((variable.value)))

ui.run()