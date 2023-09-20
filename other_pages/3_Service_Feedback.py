from utils import *
from templates.section_template_simplify import SectionTemplate

st.set_page_config(page_title='Feedback Reviewer', page_icon=':smile:', layout='wide', initial_sidebar_state='auto')
from login_script import login

config_file = "config.yaml"
login(render_func=SectionTemplate, 
      config_file=config_file, 
      section='Service')