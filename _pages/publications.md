---
layout: archive
title: "Selected Publications"
permalink: /publications/
author_profile: true
---

You can also find comprehensive list of my articles on my <u><a href="https://scholar.google.com/citations?hl=en&user=oUhThscAAAAJ">Google Scholar</a></u> profile.


{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

<sup>*</sup> Equal authorship statement
<br>
<i>[C]</i> Conference
