try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys


BLOG_DIR = os.environ['C:\Liam427.github.io']
# BLOG_DIR = '/Users/cscorley/git/cscorley.github.io/'

f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        f = arg.split('.ipynb')[0]
        break


c = get_config()
c.NbConvertApp.export_format = 'markdown'
c.MarkdownExporter.template_path = ['/Users/blahb/.ipython/templates']
c.MarkdownExporter.template_file = 'jekyll'
#c.Exporter.file_extension = 'md'

def path2support(path):
    """Turn a file path into a URL"""
    parts = path.split(os.path.sep)
    return '{{ site.baseurl}}notebooks/' + '/'.join(quote(part) for part in parts)

c.MarkdownExporter.filters = {'path2support': path2support}

if f:
    c.NbConvertApp.output_base = f.lower().replace(' ', '-')
    c.FilesWriter.build_directory = BLOG_DIR + '/_jupyter'