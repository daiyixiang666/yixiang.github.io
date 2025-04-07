---
# Leave the homepage title empty to use the site title
title: ""
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: "6rem"

sections:
  - block: resume-biography-3
    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
      text: ""
      # Show a call-to-action button under your biography? (optional)
      button:
        text: Download CV
        url: uploads/resume.pdf
    design:
      css_class: dark
      background:
        color: black
        image:
          # Add your image background to `assets/media/`.
          filename: stacked-peaks.svg
          filters:
            brightness: 1.0
          size: cover
          position: center
          parallax: false
  - block: markdown
    content:
      title: '🧪 My Research'
      subtitle: ''
      text: |-
        I'm an undergraduate researcher at the University of Michigan, focusing on generative AI—especially diffusion and autoregressive models. 

        My work explores the theoretical foundations and practical applications of image and video generation. From understanding Gaussian structures in diffusion models to accelerating training with efficient architectures, I aim to make generative models more interpretable, scalable, and impactful.

        I'm always open to collaboration or cool ideas—feel free to reach out!
    design:
      columns: '1'
  
---
