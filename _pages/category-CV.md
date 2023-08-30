---
title: "CV"
layout: archive
permalink: /CV
author_profile: true
sidebar:
    nav: "sidebar-category"
---

{% assign posts = site.categories.CV %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
