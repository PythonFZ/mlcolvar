from nicegui import ui

with ui.stepper().props('vertical').classes('w-full') as stepper:
    with ui.step('Choose data type'):
        # box to choose the data type
        ui.label('Choose the type of dataset you want to use')
        data_types = ui.radio({'Unsupervised': 'Unlabeled', 
                               'Supervised': 'Labeled', 
                               'Time-informed': 'Time-informed', 
                               'Multi-task': 'Multi-type'
                               }, 
                               value=None).props('inline')
        with ui.stepper_navigation():
                    ui.button('Next', on_click=stepper.next)



    with ui.step('Choose CV model'):

        cv_model = ui.select({'AE CV': 'AE CV',
                              'VAE CV': 'VAE CV', 
                              'Regression CV': 'Regression CV', 
                              'Deep-LDA CV': 'Deep-LDA CV', 
                              'Deep-TDA CV': 'Deep-TDA CV', 
                              'Deep-TICA CV': 'Deep-TICA CV', 
                              'Multitask CV': 'Multitask CV'}, value='EMPTY')#.set_visibility(False)
        
        # box to propose the CVs based on the data type
        # unsupervised
        with ui.column().bind_visibility_from(data_types, 'value', value='Unsupervised'):
            ui.label('With unlabeled dataset you can use unsupervised learning. The options are:')
            aux = ui.radio(options = {'AE CV': 'AE CV', 'VAE CV': 'VAE CV'}, value='miao').props('inline').bind_value_to(cv_model, 'value')

        # supervised
        with ui.column().bind_visibility_from(data_types, 'value', value='Supervised'):
            ui.label('With labeled dataset you can use supervised learning. The options are:')
            aux = ui.radio(options = {1: 'Regression CV', 2: 'Deep-LDA CV', 3: 'Deep-TDA CV'}, value='miao').props('inline').bind_value_to(cv_model, 'value')

        # # time-lagged
        # with ui.column().bind_visibility_from(data_types, 'value', value='Time-informed'):
        #     ui.label('With Time-lagged dataset you can use time-informed learning. The options are:')
        #     cv_model = ui.radio(options = {1: 'Deep-TICA CV'}).props('inline')

        # # multi task
        # with ui.column().bind_visibility_from(data_types, 'value', value='Multi-task'):
        #     ui.label('With multiple datasets you can use multi-task learning. The options are:')
        #     cv_model = ui.radio(options = {1: 'Multitask CV'}).props('inline')

        with ui.stepper_navigation():
            ui.button('Next', on_click=stepper.next)
            ui.button('Back', on_click=stepper.previous).props('flat')


    with ui.step('Set Cv parameters'):
        # ui.button(f'miao {unsupervised.value} {supervised.value}')

        ui.button('Check cv model', on_click=lambda: ui.label(cv_model.value))

        # # test set params for TDA
        # with ui.column().bind_visibility_from(cv_method, 'value', value=3):
        #     # TODO maybe print docstring
        #     ui.label(f'You chose {cv_method.value}. The options of this method are:')
        
        with ui.stepper_navigation():
            ui.button('Next', on_click=stepper.next)        
            ui.button('Back', on_click=stepper.previous).props('flat')




ui.run()