from nicegui import ui


# proxy for the choice of the cv model
class CV_METHOD():
    def __init__(self):
        self.value=None
        
    def set_value(self, value):
         self.value = value

    def return_value(self):
        return self.value


# util to skip to next step
def end_step(next=True, back=True):
    with ui.stepper_navigation():
        if next: ui.button('Next', on_click=stepper.next)
        if back: ui.button('Back', on_click=stepper.previous).props('flat')

def center_selection(n_centers):
    with ui.grid(columns=n_centers):
        for i in range(n_centers):
            ui.number(label='centers', 
                            value=0, 
                            format='%f',
                            )

def update_variable(variable, value):
    variable = value
    return variable




with ui.stepper().props('vertical').classes('w-full') as stepper:
    
# 1) box to choose the data type
    with ui.step('Choose data type'):
        ui.label('Choose the type of dataset you want to use')
        data_types = ui.radio({'Unsupervised': 'Unlabeled', 
                               'Supervised': 'Labeled', 
                               'Time-informed': 'Time-informed', 
                               'Multi-task': 'Multi-type'
                               }, 
                               value=None).props('inline')

        end_step(back=False)


# 2) box to propose the CVs based on the data type
    with ui.step('Choose CV model'):
        # initialize proxy for cv_method
        cv_method = CV_METHOD()

        # unsupervised
        with ui.column().bind_visibility_from(data_types, 'value', value='Unsupervised'):
            ui.label('With unlabeled dataset you can use unsupervised learning. The options are:')
            unsupervised = ui.radio(options = {'AE CV': 'AE CV', 
                                               'VAE CV': 'VAE CV'}, 
                                               on_change=lambda: cv_method.set_value(unsupervised.value)
                                    ).props('inline')

        # supervised
        with ui.column().bind_visibility_from(data_types, 'value', value='Supervised'):
            ui.label('With labeled dataset you can use supervised learning. The options are:')
            supervised = ui.radio(options = {'Regression CV': 'Regression CV', 
                                             'Deep-LDA CV': 'Deep-LDA CV', 
                                             'Deep-TDA CV': 'Deep-TDA CV'}, 
                                             on_change=lambda: cv_method.set_value(supervised.value), 
                                    ).props('inline')

        # time-lagged
        with ui.column().bind_visibility_from(data_types, 'value', value='Time-informed'):
            ui.label('With Time-lagged dataset you can use time-informed learning. The options are:')
            time_informed = ui.radio(options = {'Deep-TICA CV': 'Deep-TICA CV'}, 
                                               on_change=lambda: cv_method.set_value(time_informed.value)
                                    ).props('inline')

        # multi task
        with ui.column().bind_visibility_from(data_types, 'value', value='Multi-task'):
            ui.label('With multiple datasets you can use multi-task learning. The options are:')
            multi_task = ui.radio(options = {'Multitask CV': 'Multitask CV'}, 
                                            on_change=lambda: cv_method.set_value(multi_task.value)
                                    ).props('inline')

        end_step()


# 3) box to set the parameters
    with ui.step('Set CV parameters'):
        ui.update()
        ui.button('Check CV model', on_click=lambda: ui.label(cv_method.value))
        ui.label().bind_text_from(cv_method, 'value', backward=lambda value: f'You chose {value}. The options of this method are:')


        # test set params for TDA
        with ui.column().bind_visibility_from(cv_method, 'value', value='AE CV'):
            # TODO maybe print docstring
            encoder = ui.input(label='Encoder layers')
            decoder = ui.input(label='Decoder layers')
            
                    
        end_step()

    with ui.step('Set CV parameters'):
        ui.button('Check CV model', on_click=lambda: ui.label(encoder.value)).bind_visibility_from(cv_method, 'value', value='AE CV')


ui.run()

# n_cvs =  ui.number(label='n_cvs', 
            #                    value=1, 
            #                    format='%d',
            #                    min=1, 
            #                    validation={'You can do better!': lambda value: value < n_states.value}
            #                   )