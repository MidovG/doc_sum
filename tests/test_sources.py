from docsum.sources import drive_folder_id, is_drive_url


def test_drive_url_parsing():
    url = "https://drive.google.com/drive/folders/1x6EKNkVw6PlFVTr6cGrsVscmRuwqGrXd?usp=sharing"
    assert is_drive_url(url)
    assert drive_folder_id(url) == "1x6EKNkVw6PlFVTr6cGrsVscmRuwqGrXd"
    assert not is_drive_url("/home/user/docs")
