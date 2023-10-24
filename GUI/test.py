from nicegui import ui

ui.label('Hello NiceGUI!')

class Demo:
    def __init__(self):
        self.number = 1

def test(x):
        x = x + 10
        return x


demo = Demo()
v = ui.checkbox('visible', value=True)
with ui.column().bind_visibility_from(v, 'value'):
    ui.slider(min=1, max=3).bind_value(demo, 'number')
    ui.toggle({1: 'A', 2: 'B', 3: 'C'}).bind_value(demo, 'number')
    ui.number().bind_value(demo, 'number')
    ui.button('Click me!', on_click=lambda: ui.label(test(10)))

ui.run()