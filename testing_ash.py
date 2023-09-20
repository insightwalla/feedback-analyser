import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['Bunmaska1']).generate()
print(hashed_passwords)