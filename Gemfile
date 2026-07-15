# frozen_string_literal: true

source "https://rubygems.org"

# Force Ruby version constraint flexibility to avoid conflict with html-proofer 5.0+
ruby "~> 3.1"

gem "jekyll-theme-chirpy", "~> 7.6"

gem "html-proofer", "~> 5.0", group: :test

# Corrected valid platform identifiers for Bundler compatibility
platforms :mingw, :mswin, :x64_mingw, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :mswin, :x64_mingw, :jruby]
