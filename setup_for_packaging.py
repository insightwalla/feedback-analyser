import wx # import wxPython
# need webkit for the browser
import wx.html2 as webview

# create a window that opena a web browser
class MyBrowser(wx.Frame):
      def __init__(self, parent, title):
         # start the streamlit server
         wx.Frame.__init__(self, parent, title=title, size=(800,600))
         self.browser = webview.WebView.New(self)
         self.browser.LoadURL("https://insightwalla-feedback-review-helper-1-sentiment-analysis-wt761j.streamlit.app")
         self.Show()

# create the app
app = wx.App()
# create the window
frame = MyBrowser(None, "Dishoom ~ Feedback Reviewer")
# start the app
app.MainLoop()