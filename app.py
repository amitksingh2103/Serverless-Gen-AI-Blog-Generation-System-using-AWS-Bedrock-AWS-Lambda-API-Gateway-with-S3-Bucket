import streamlit as st
import requests

st.set_page_config(page_title="Blog Generator", layout="centered")

st.title("📝 Blog Generator (AWS Bedrock Serverless)")

API_URL = "https://<API-ENDPOINT>"

topic = st.text_input("Enter topic for blog")

if st.button("Generate"):
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    with st.spinner("Generating..."):
        try:
            response = requests.post(API_URL, json={"blog_topic": topic}, timeout=60)

            if response.status_code == 200:
                data = response.json()
                
                st.success("Blog generation completed")

                s3_uri = None
                if isinstance(data, dict):
                    s3_uri = data.get("s3_uri") or f"s3://{data.get('s3_bucket')}/{data.get('s3_key')}"
                
                if s3_uri:
                    st.write("Stored at:")
                    st.code(s3_uri)

                blog = data.get("blog")
                if blog:
                    st.write("Generated Blog:")
                    st.markdown(blog, unsafe_allow_html=True)
                else:
                    st.info("Blog saved in S3 and not displayed here.")
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(str(e))


