from utils import *
from templates.section_final_counter_ import final_page
st.set_page_config(page_title='Feedback Reviewer', page_icon=':smile:', layout='wide', initial_sidebar_state='auto')
from login_script import login

config_file = "config.yaml"
login(render_func=final_page, 
      config_file=config_file, 
      section='')