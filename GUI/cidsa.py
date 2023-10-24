from nicegui import ui

container = ui.row()

def add_face():
    with container:
        col = ui.column()
        with col:
            n_states = ui.number(label='States', 
                                    value=5, 
                                    format='%d',
                                    min=2,
                                    )
            
            n_states = ui.number(label='States', 
                                    value=5, 
                                    format='%d',
                                    min=2,
                                    )
add_face()

ui.button('Add', on_click=add_face)
ui.button('Remove', on_click=lambda: container.remove(0) if list(container) else None)
ui.button('Clear', on_click=container.clear)

ui.run()